### Decoding saved messages for a thread (SQLite checkpointer)

This guide shows how to inspect LangGraph persistence tables and decode message history for a given `thread_id` from the SQLite database at `db/db.db`.

#### 1) Inspect tables and recent threads

```bash
sqlite3 db/db.db ".tables"
sqlite3 db/db.db "SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC LIMIT 10;"
```

Tables of interest:
- `checkpoints` — snapshot of full state per checkpoint
- `writes` — per-channel incremental writes (msgpack payloads), including `channel='messages'`

Schemas (for reference):

```sql
PRAGMA table_info(checkpoints);
-- 0 thread_id TEXT, 1 checkpoint_ns TEXT, 2 checkpoint_id TEXT, 3 parent_checkpoint_id TEXT,
-- 4 type TEXT, 5 checkpoint BLOB, 6 metadata BLOB

PRAGMA table_info(writes);
-- 0 thread_id TEXT, 1 checkpoint_ns TEXT, 2 checkpoint_id TEXT, 3 task_id TEXT,
-- 4 idx INTEGER, 5 channel TEXT, 6 type TEXT, 7 value BLOB
```

#### 2) Quick SQL peek for a specific thread

Replace `<THREAD_ID>` as needed:

```bash
sqlite3 db/db.db "SELECT checkpoint_ns, checkpoint_id, length(checkpoint) FROM checkpoints WHERE thread_id='<THREAD_ID>' ORDER BY rowid DESC LIMIT 3;"
sqlite3 db/db.db "SELECT checkpoint_ns, checkpoint_id, idx, channel, type, length(value) FROM writes WHERE thread_id='<THREAD_ID>' AND channel='messages' ORDER BY rowid DESC LIMIT 5;"
```

#### 3) Python snippet to decode messages from a checkpoint

Messages are stored as msgpack with custom types. Use an `ext_hook` to unpack message objects and extract readable fields.

```python
import sqlite3, msgpack
from pathlib import Path

DB = Path('db/db.db')
THREAD_ID = '<THREAD_ID>'  # set me

def decode_ext(code, data):
    # LangChain/LangGraph encode objects (e.g., HumanMessage/AIMessage) as ExtType code 5.
    if code == 5:
        try:
            # ext payload is itself msgpack: [module, class, payload_dict, ...]
            obj = msgpack.unpackb(data, strict_map_key=False, ext_hook=decode_ext)
            if isinstance(obj, (list, tuple)) and len(obj) >= 3:
                module, cls, payload = obj[0], obj[1], obj[2]
                if isinstance(payload, dict):
                    payload['__class__'] = cls
                return payload
            return obj
        except Exception:
            return msgpack.ExtType(code, data)
    return msgpack.ExtType(code, data)

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
cur = con.cursor()

row = cur.execute(
    """
    SELECT checkpoint
    FROM checkpoints
    WHERE thread_id=?
    ORDER BY rowid DESC LIMIT 1
    """,
    (THREAD_ID,)
).fetchone()

if not row:
    print('No checkpoint found for thread')
    raise SystemExit(0)

chk = msgpack.unpackb(row['checkpoint'], strict_map_key=False, ext_hook=decode_ext)
msgs = []
if isinstance(chk, dict):
    chvals = chk.get('channel_values', {})
    msgs = chvals.get('messages', []) or []

print(f"Decoded checkpoint messages: {len(msgs)}")
for i, m in enumerate(msgs):
    if isinstance(m, dict):
        cls = m.get('__class__') or m.get('type')
        content = m.get('content')
        print(f"{i}. {cls}: {str(content)[:200]}")
    else:
        print(f"{i}. {type(m)} {str(m)[:200]}")
```

#### 4) Python snippet to reconstruct messages via writes channel

Use `writes` rows for `channel='messages'`. Each row’s `value` is a msgpack payload (often a list with one message). Iterate in ascending order to rebuild the timeline.

```python
import sqlite3, msgpack
from pathlib import Path

DB = Path('db/db.db')
THREAD_ID = '<THREAD_ID>'  # set me

def decode_ext(code, data):
    if code == 5:
        try:
            obj = msgpack.unpackb(data, strict_map_key=False, ext_hook=decode_ext)
            if isinstance(obj, (list, tuple)) and len(obj) >= 3:
                module, cls, payload = obj[0], obj[1], obj[2]
                if isinstance(payload, dict):
                    payload['__class__'] = cls
                return payload
            return obj
        except Exception:
            return msgpack.ExtType(code, data)
    return msgpack.ExtType(code, data)

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
cur = con.cursor()

rows = cur.execute(
    """
    SELECT checkpoint_id, idx, value
    FROM writes
    WHERE thread_id=? AND channel='messages'
    ORDER BY rowid ASC
    """,
    (THREAD_ID,)
).fetchall()

history = []
for r in rows:
    data = msgpack.unpackb(r['value'], strict_map_key=False, ext_hook=decode_ext)
    if isinstance(data, list):
        for m in data:
            if isinstance(m, dict):
                history.append({
                    'checkpoint_id': r['checkpoint_id'],
                    'idx': r['idx'],
                    'class': m.get('__class__') or m.get('type'),
                    'content': m.get('content')
                })

print(f"Reconstructed messages via writes: {len(history)}")
for i, m in enumerate(history):
    print(f"{i}. [{m['checkpoint_id']}#{m['idx']}] {m['class']}: {str(m['content'])[:200]}")
```

#### Notes
- Use the same `THREAD_ID` you run the graph with (`configurable.thread_id`).
- Checkpoint snapshots reflect the merged state at that point; writes are the granular per-channel updates.
- For Postgres, the approach is analogous (same tables/columns under the Postgres checkpointer), but run SQL against your Postgres URI.
- If you see `ExtType(code=5, ...)` when printing values directly, use the `ext_hook` approach above to unpack into readable dicts.

