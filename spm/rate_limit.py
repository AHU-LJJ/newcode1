"""Persistent per-model rolling TPM admission, shared by all API roles."""
import asyncio
import math
import time


class TokenCounter:
    def __init__(self, tokenizer_path=None):
        self.tokenizer = None
        if tokenizer_path:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def __call__(self, payload):
        # Qwen's ChatML text format; extra allowance covers provider framing.
        if 'messages' not in payload:
            inputs=payload.get('input',[])
            inputs=[inputs] if isinstance(inputs,str) else inputs
            return sum(len(str(x).encode('utf-8')) for x in inputs)+256
        messages = payload.get('messages', [])
        text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        text += '<|im_start|>assistant\n'
        count = (len(self.tokenizer.encode(text, add_special_tokens=False).ids)
                 if self.tokenizer else len(text.encode('utf-8')))
        return count + int(payload.get('max_tokens', 0)) + 256


class WindowLimiter:
    def __init__(self, db, limits, headroom=.9, lease_seconds=420,
                 clock=time.time, sleep=asyncio.sleep):
        if not 0 < headroom <= 1 or any(type(v) is not int or v <= 0 for v in limits.values()):
            raise ValueError('TPM limits must be positive integers; headroom must be in (0, 1]')
        self.db, self.clock, self.sleep = db, clock, sleep
        self.capacity = {k: math.floor(v*headroom) for k,v in limits.items()}
        self.lease_seconds = max(60, lease_seconds)
        self.wait_seconds = 0.
        self.reservations = 0
        self.underestimated = 0
        db.execute('CREATE TABLE IF NOT EXISTS tpm_reservations '
                   '(id INTEGER PRIMARY KEY, service TEXT, tokens INTEGER, expires REAL)')
        db.commit()

    async def acquire(self, service, model, tokens):
        if model not in self.capacity:return None
        capacity = self.capacity[model]
        if tokens > capacity:
            raise ValueError('Single request exceeds configured TPM capacity; reduce request/output size')
        while True:
            now = self.clock()
            self.db.execute('BEGIN IMMEDIATE')
            try:
                self.db.execute('DELETE FROM tpm_reservations WHERE expires<=?', (now,))
                used, next_expiry = self.db.execute(
                    'SELECT COALESCE(SUM(tokens),0), MIN(expires) FROM tpm_reservations WHERE service=?',
                    (service,)).fetchone()
                if used+tokens <= capacity:
                    reservation = self.db.execute('INSERT INTO tpm_reservations(service,tokens,expires) VALUES (?,?,?)',
                        (service,tokens,now+self.lease_seconds)).lastrowid
                    self.db.commit();self.reservations += 1
                    return reservation
                delay = min(1., max(.01, next_expiry-now))
                self.db.commit()
            except BaseException:
                self.db.rollback();raise
            started = self.clock()
            await self.sleep(delay)
            self.wait_seconds += max(0., self.clock()-started)

    def finish(self, reservation, known_tokens=None):
        if reservation is None:return
        row = self.db.execute('SELECT tokens FROM tpm_reservations WHERE id=?', (reservation,)).fetchone()
        if row is None:return
        tokens = row[0]
        if type(known_tokens) is int and known_tokens >= 0:
            self.underestimated += int(known_tokens > tokens)
            # Refund unused completion allowance, retaining framing headroom.
            tokens = known_tokens+256
        # Keep completed usage for a full minute after completion. Unknown usage
        # retains its reservation; abandoned requests expire after their lease.
        self.db.execute('UPDATE tpm_reservations SET tokens=?,expires=? WHERE id=?',
                        (tokens,self.clock()+60,reservation))
        self.db.commit()

    def usage(self):
        return dict(capacity=self.capacity, reservations=self.reservations,
                    summed_request_wait_seconds=self.wait_seconds,
                    underestimated_requests=self.underestimated,
                    scope='This cache database; all roles sharing a provider/model. Usage retained 60s after completion.')
