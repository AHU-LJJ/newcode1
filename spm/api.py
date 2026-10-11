"""Async API transport, validated cache, in-flight coalescing and persistent accounting."""
import asyncio
from collections import defaultdict
from contextvars import ContextVar
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import random
import sqlite3
import time

import httpx
import numpy as np
from pydantic import ValidationError

from .models import fingerprint
from .rate_limit import TokenCounter, WindowLimiter


class APIError(RuntimeError):
    pass


class BudgetExceeded(APIError):
    pass


class AccountUnavailable(APIError):
    pass


REQUEST_CONTEXT = ContextVar('spm_request_context', default={})


class Gateway:
    def __init__(self, endpoints, cache_path, concurrency=16, max_calls=2000,
                 max_tokens=5_000_000, timeout=90, retries=5, max_output=4096,
                 tpm_limits=None, tpm_tokenizer=None):
        self.endpoints = endpoints
        self.sem = asyncio.Semaphore(concurrency)
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=20),
            follow_redirects=False, limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency))
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, run TEXT, role TEXT, tag TEXT, json TEXT)")
        self.db.commit()
        self.run = fingerprint([datetime.now(timezone.utc).isoformat(), id(self)])[:20]
        self.inflight = {}
        self.cooldowns = defaultdict(float)
        self.payment_blocked = set()
        self.hits = defaultdict(int)
        self.coalesced = defaultdict(int)
        self.schema_failures = defaultdict(int)
        self.calls = 0
        self.tokens_bound = 0
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.retries, self.max_output = retries, max_output
        # SDK callers receive the same Qwen protection as the experiment CLI.
        qwen='Qwen/Qwen2.5-7B-Instruct'
        if tpm_limits is None and any(e.model==qwen for e in endpoints.values()):
            tpm_limits={qwen:80000}
        local_tokenizer=Path(__file__).resolve().parents[1]/'datasets/tokenizers/qwen2.5-7b/tokenizer.json'
        if tpm_limits and qwen in tpm_limits and not tpm_tokenizer and local_tokenizer.exists():
            tpm_tokenizer=str(local_tokenizer)
        self.token_counter = TokenCounter(tpm_tokenizer) if tpm_limits else None
        self.limiter = WindowLimiter(self.db,tpm_limits,lease_seconds=2*timeout+60) if tpm_limits else None

    async def close(self):
        pending=list(self.inflight.values())
        for task in pending:
            if not task.done():task.cancel()
        await asyncio.gather(*pending,return_exceptions=True)
        usage=self.usage()
        await self.http.aclose()
        self.db.close()
        return usage

    def _reserve(self, payload):
        # Conservative reservation BEFORE issuing a request, including all retries.
        reserve = len(json.dumps(payload, ensure_ascii=False).encode()) + payload.get("max_tokens", 0) + 1024
        if self.calls >= self.max_calls or self.tokens_bound + reserve > self.max_tokens:
            raise BudgetExceeded("SPM API budget exhausted before request; use cache/resume or a larger explicit budget")
        self.calls += 1
        self.tokens_bound += reserve
        return reserve

    def _record(self, role, tag, record, reserve, record_id):
        usage = record.get("usage", {})
        known = usage.get("total_tokens")
        if type(known) is int and known >= 0:
            self.tokens_bound += known - reserve
        # Unknown usage keeps its conservative reservation, avoiding false free retries.
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        if record.get('started_at'):
            record['elapsed_seconds']=max(0.,(datetime.fromisoformat(record['timestamp'])-datetime.fromisoformat(record['started_at'])).total_seconds())
        record['reserved_tokens']=reserve
        record['experiment_context']=dict(REQUEST_CONTEXT.get())
        self.db.execute('UPDATE requests SET json=? WHERE id=?',(json.dumps(record),record_id))
        self.db.commit()

    @asynccontextmanager
    async def _admission(self, service, model, payload):
        # A throttled Qwen call must not occupy an HTTP slot needed by BGE/judge.
        reservation=None
        if self.limiter and model in self.limiter.capacity:
            reservation=await self.limiter.acquire(service,model,self.token_counter(payload))
        try:
            await self.sem.acquire()
        except BaseException:
            if self.limiter:self.limiter.finish(reservation,0)
            raise
        try:
            yield reservation
        finally:
            self.sem.release()

    async def _request(self, role, path, payload, tag):
        endpoint = self.endpoints[role]
        service = fingerprint([endpoint.base_url, endpoint.model])
        billing_service=fingerprint([endpoint.base_url,endpoint.model,endpoint.key])
        def check_payment():
            if billing_service in self.payment_blocked:
                raise AccountUnavailable('API HTTP 402 previously received for this model/account; restore API credit and start a new invocation')
        for attempt in range(self.retries):
            check_payment()
            remaining = self.cooldowns[service] - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
            record = dict(attempt=attempt + 1, usage={}, usage_complete=False)
            retry, delay = False, min(30, 2 ** attempt) + random.random()
            async with self._admission(service,endpoint.model,payload) as rate_reservation:
                # Waiting requests must recheck after another request reports 402.
                check_payment()
                try:
                    check_payment()
                    reserve = self._reserve(payload)
                except BaseException:
                    if self.limiter:self.limiter.finish(rate_reservation,0)
                    raise
                record['started_at']=datetime.now(timezone.utc).isoformat()
                provisional=dict(record,error='Request in flight or process interrupted',reserved_tokens=reserve,
                                 timestamp=datetime.now(timezone.utc).isoformat())
                record_id=self.db.execute('INSERT INTO requests(run,role,tag,json) VALUES (?,?,?,?)',
                                          (self.run,role,tag,json.dumps(provisional))).lastrowid
                self.db.commit()
                try:
                    response = await self.http.post(endpoint.base_url.rstrip("/") + path,
                        json=payload, headers={"Authorization": "Bearer " + endpoint.key})
                    record["http_status"] = response.status_code
                    if response.status_code == 429 or response.status_code >= 500:
                        retry = True
                        record["error"] = f"API HTTP {response.status_code}"
                        delay = max(delay, 8 if response.status_code == 429 else 2)
                        value = response.headers.get("Retry-After")
                        if value:
                            try:
                                delay = max(delay, float(value))
                            except ValueError:
                                try:
                                    delay = max(delay, parsedate_to_datetime(value).timestamp() - time.time())
                                except (ValueError, TypeError, OverflowError):
                                    pass
                        delay = min(60, delay)
                        self.cooldowns[service] = max(self.cooldowns[service], time.monotonic() + delay)
                    elif response.is_error or response.is_redirect:
                        record["error"] = f"API HTTP {response.status_code}"
                        if response.status_code==402:self.payment_blocked.add(billing_service)
                    else:
                        data = response.json()
                        if not isinstance(data, dict):
                            raise ValueError("Expected response object")
                        usage = data.get("usage", {})
                        if isinstance(usage, dict):
                            record["usage"] = usage
                            # Embeddings commonly omit completion_tokens.
                            record["usage_complete"] = type(usage.get("total_tokens")) is int
                        record["response_model"] = data.get("model")
                        self._record(role, tag, record, reserve,record_id)
                        return data
                except asyncio.CancelledError:
                    record['error']='Request cancelled; usage unknown'
                    self._record(role,tag,record,reserve,record_id)
                    raise
                except httpx.TransportError:
                    record["error"] = "API transport failure (credentials and URL omitted)"
                    retry = True
                except (ValueError, TypeError):
                    record["error"] = "API returned invalid JSON object"
                    retry = True
                finally:
                    if self.limiter:self.limiter.finish(rate_reservation,record.get('usage',{}).get('total_tokens'))
                self._record(role, tag, record, reserve,record_id)
            if not retry or attempt + 1 == self.retries:
                if billing_service in self.payment_blocked:
                    raise AccountUnavailable(record.get('error','API HTTP 402')) from None
                raise APIError(record.get("error", "API request failed")) from None
            await asyncio.sleep(delay)
        raise APIError("API retries exhausted")

    async def _cached(self, key, tag, fn):
        row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            self.hits[tag] += 1
            return json.loads(row[0])
        if key in self.inflight:
            self.coalesced[tag] += 1
            return await asyncio.shield(self.inflight[key])
        async def perform():
            result = await fn()
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(result)))
            self.db.commit()
            return result
        task = asyncio.create_task(perform())
        self.inflight[key] = task
        def cleanup(done):
            if self.inflight.get(key) is done:self.inflight.pop(key,None)
            if not done.cancelled():done.exception()
        task.add_done_callback(cleanup)
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self.inflight.pop(key, None)

    async def ask(self, role, system, value, schema=None, tag="misc", validate=None, max_tokens=None):
        endpoint = self.endpoints[role]
        original = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        instructions = system
        if schema is not None:
            instructions += "\nReturn only a JSON object with this schema:\n" + json.dumps(schema.model_json_schema())
        payload = dict(model=endpoint.model, temperature=0, max_tokens=max_tokens or self.max_output,
                       messages=[dict(role="system", content=instructions), dict(role="user", content=original)])
        if "deepseek-r1" not in endpoint.model.lower():
            payload["enable_thinking"] = False
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        key = fingerprint([endpoint.base_url, payload, tag])
        async def fetch():
            request = dict(payload)
            for repair in range(3):
                choice={}
                data = await self._request(role, "/chat/completions", request, tag)
                try:
                    choice = data["choices"][0]
                    text = choice["message"]["content"]
                    if choice.get('finish_reason')=='length':raise ValueError('Output token limit reached')
                    if choice.get("finish_reason") != "stop" or not isinstance(text, str) or not text.strip():
                        raise ValueError("Incomplete model response")
                    if schema is None:
                        return text
                    obj = schema.model_validate_json(text)
                    if validate:
                        validate(obj)
                    return obj.model_dump(mode="json")
                except (ValueError, ValidationError, KeyError, IndexError, TypeError) as error:
                    self.schema_failures[tag]+=1
                    # ValidationError details may contain all source text; keep feedback short.
                    details = (str(error)[:500] if not isinstance(error, ValidationError)
                               else str([dict(loc=e['loc'], type=e['type'],msg=e['msg'][:240])
                                         for e in error.errors(include_input=False,include_url=False)[:3]])[:1000])
                    if repair == 2:
                        raise APIError(f"{tag}: invalid model response after three schema attempts ({details})") from None
                    output_limit=request['max_tokens']
                    if str(error)=='Output token limit reached':output_limit=min(16384,output_limit*2)
                    previous=choice.get('message',{}).get('content')
                    feedback='Correct the JSON. Validation error: '+details
                    if tag in ('extract','normalize'):
                        if schema.__name__=='SourceExtraction':
                            feedback+=' Select only a supplied USER source_id. The host binds literal evidence. Do not add span or turn_idx. Omit unsupported extracted units.'
                        elif schema.__name__=='PreferenceFields':
                            feedback+=' Normalize this one supplied known preference. The host binds literal evidence. Do not add span, turn_idx or expression.'
                        else:
                            feedback+=' Use a SHORT exact continuous quote from the supplied USER turn. Preserve original Unicode, capitalization and punctuation. Do not combine non-contiguous fragments or paraphrase. Omit unsupported extracted units instead of inventing evidence.'
                    repair_messages=payload['messages']+([dict(role='assistant',content=previous)] if isinstance(previous,str) and previous.strip() else [])
                    request = dict(payload,max_tokens=output_limit,messages=repair_messages+[
                        dict(role="user", content=feedback)])
            raise APIError("Unreachable schema retry")
        result = await self._cached(key, tag, fetch)
        obj = schema.model_validate(result) if schema else result
        if schema and validate:
            validate(obj)
        return obj

    async def embed(self, texts, batch=32):
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        endpoint = self.endpoints["embed"]
        keys = [fingerprint([endpoint.base_url, endpoint.model, text]) for text in texts]
        unique = dict(zip(keys, texts))
        vectors, todo = {}, []
        for key, text in unique.items():
            row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            if row:
                vectors[key] = json.loads(row[0]); self.hits["embedding"] += 1
            else:
                todo.append((key, text))
        async def encode(items):
            # Deduplicate concurrent identical batches as well as individual cache hits.
            key = fingerprint(["embedding_batch", [k for k, _ in items]])
            async def fetch():
                data = await self._request("embed", "/embeddings",
                    dict(model=endpoint.model, input=[t for _, t in items]), "embedding")
                rows = data.get("data", [])
                if sorted(r.get("index", -1) for r in rows) != list(range(len(items))):
                    raise APIError("Embedding API returned missing or duplicate indices")
                ordered = sorted(rows, key=lambda r: r["index"])
                matrix = np.asarray([r["embedding"] for r in ordered], dtype=np.float32)
                if matrix.shape != (len(items), 1024) or not np.isfinite(matrix).all():
                    raise APIError("Expected finite 1024-dimensional BGE-M3 vectors")
                norms = np.linalg.norm(matrix, axis=1)
                if (norms == 0).any():
                    raise APIError("Zero embedding vector")
                return (matrix / norms[:, None]).tolist()
            result = await self._cached(key, "embedding", fetch)
            for (item_key, _), vector in zip(items, result, strict=True):
                # Different overlapping batches can produce slightly different provider
                # vectors. Commit one canonical vector per text; every caller reads
                # the winner, including callers whose batch completed later.
                self.db.execute("INSERT OR IGNORE INTO cache VALUES (?,?)", (item_key, json.dumps(vector)))
                vectors[item_key] = json.loads(self.db.execute('SELECT value FROM cache WHERE key=?',(item_key,)).fetchone()[0])
            self.db.commit()
        await asyncio.gather(*(encode(todo[i:i + batch]) for i in range(0, len(todo), batch)))
        result = np.asarray([vectors[k] for k in keys], dtype=np.float32)
        if result.shape != (len(texts), 1024) or not np.isfinite(result).all() or (np.linalg.norm(result, axis=1) == 0).any():
            raise APIError("Invalid cached BGE-M3 vectors")
        return result

    def usage(self,run_id=None):
        requested=run_id or self.run
        groups = defaultdict(lambda: dict(requests=0, errors=0, prompt_tokens=0, completion_tokens=0,
                                         total_tokens=0, usage_complete=True))
        count=upper_bound=unknown=0;latencies=defaultdict(list)
        for role, tag, raw in self.db.execute("SELECT role,tag,json FROM requests WHERE run=?", (requested,)):
            data = json.loads(raw); group = groups[role + ":" + tag]
            count+=1
            known=data.get('usage',{}).get('total_tokens')
            unknown+=int(type(known) is not int)
            upper_bound+=known if type(known) is int else data.get('reserved_tokens',0)
            group["requests"] += 1; group["errors"] += int("error" in data)
            if data.get('elapsed_seconds') is not None:latencies[role+':'+tag].append(data['elapsed_seconds'])
            group["usage_complete"] &= data["usage_complete"]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                group[k] += data.get("usage", {}).get(k, 0)
        for tag,seconds in latencies.items():
            groups[tag]['http_latency']=dict(measured_requests=len(seconds),total_seconds=sum(seconds),
                median_seconds=float(np.median(seconds)),p95_seconds=float(np.quantile(seconds,.95)))
        current=requested==self.run
        return dict(run_id=requested, by_tag=dict(groups), cache_hits=dict(self.hits) if current else None,coalesced=dict(self.coalesced) if current else None,
                    tpm_throttle=self.limiter.usage() if current and self.limiter else None,
                    schema_failures=dict(self.schema_failures) if current else None,
                    http_requests=self.calls if current else count,tokens_upper_bound=self.tokens_bound if current else upper_bound,
                    recovered_from_ledger=not current,
                    unknown_usage_requests=unknown,
                    known_tokens=sum(v["total_tokens"] for v in groups.values()))
