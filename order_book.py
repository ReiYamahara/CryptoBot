import asyncio
import json
import zlib
import time
import os
import pandas as pd
from datetime import datetime
from sortedcontainers import SortedDict
import websockets
from decimal import Decimal

# THIS FILE TAKES IN L2 ORDER BOOK SNAPSHOT AND UPDATES TO GENERATE A LIVE INTERNAL ORDER BOOK

DATA_DIR = "kraken_data"
BATCH_SIZE = 100 # Change to 10000 when actually running

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class ChecksumError(Exception):
    pass

class DataRecorder:
    def __init__(self, symbol):
        self.symbol = symbol.replace("/", "_") # e.g. BTC_USD
        self.buffer = []
        self.file_counter = 0
        
    def log_snapshot(self, snapshot_data):
        # Log all bids
        ts = time.time()
        for entry in snapshot_data['bids']:
            self.buffer.append({
                'timestamp': ts,
                'symbol': self.symbol,
                'event': 'snapshot',
                'side': 'bid',
                'price': str(entry['price']),
                'qty': str(entry['qty'])
            })
        # Log all asks
        for entry in snapshot_data['asks']:
            self.buffer.append({
                'timestamp': ts,
                'symbol': self.symbol,
                'event': 'snapshot',
                'side': 'ask',
                'price': str(entry['price']),
                'qty': str(entry['qty'])
            })
        
        # Check if we need to flush immediately (snapshots are big)
        if len(self.buffer) >= BATCH_SIZE:
            asyncio.create_task(self.flush_buffer())

    def log_update(self, update_data):
        ts = time.time()
        # Updates can have bids, asks, or both
        for side in ['bids', 'asks']:
            if side in update_data:
                for entry in update_data[side]:
                    self.buffer.append({
                        'timestamp': ts,
                        'symbol': self.symbol,
                        'event': 'update',
                        'side': side[:-1], # remove 's' -> bid/ask
                        'price': str(entry['price']),
                        'qty': str(entry['qty'])
                    })
        
        if len(self.buffer) >= BATCH_SIZE:
            # We run this as a task so it doesn't block the next websocket message
            asyncio.create_task(self.flush_buffer())

    async def flush_buffer(self):
        if not self.buffer:
            return

        # 1. Swap buffer reference immediately so new updates go to a new list
        #    while we save the old one.
        data_to_save = self.buffer
        self.buffer = []
        
        print(f"Flushing {len(data_to_save)} records to disk...")
        
        # 2. Run the heavy I/O in a separate thread
        await asyncio.to_thread(self._write_to_parquet, data_to_save)

    def _write_to_parquet(self, data):
        try:
            df = pd.DataFrame(data)
            
            # Filename: symbol_timestamp_chunk.parquet
            filename = f"{self.symbol}_{int(time.time())}_{self.file_counter}.parquet"
            filepath = os.path.join(DATA_DIR, filename)
            
            # Use 'pyarrow' engine for speed. Compression 'snappy' is fast.
            df.to_parquet(filepath, engine='pyarrow', compression='snappy')
            
            print(f"Saved: {filename}")
            self.file_counter += 1
        except Exception as e:
            print(f"ERROR saving data: {e}")

class ChecksumError(Exception):
    pass

class OrderBookManager:
    def __init__(self, symbol="BTC/USD", depth=25, recorder=None):
        self.symbol = symbol
        self.depth = depth
        self.bids = SortedDict()
        self.asks = SortedDict()
        self.is_synced = False
        self.recorder = recorder
        
    def handle_message(self, message):
        msg_type = message.get("type")
        data = message.get("data", [])

        if not data: return

        if msg_type == "snapshot":
            print("Snapshot received.")
            # 1. Record Data
            if self.recorder:
                self.recorder.log_snapshot(data[0])
            # 2. Update State
            self.apply_snapshot(data[0])
            
        elif msg_type == "update":
            if self.is_synced:
                # 1. Record Data
                if self.recorder:
                    self.recorder.log_update(data[0])
                # 2. Update State
                self.apply_update(data[0])
        
    def apply_snapshot(self, snapshot):
        self.bids.clear()
        self.asks.clear()
        for entry in snapshot['bids']:
            self.update_entry(self.bids, entry['price'], entry['qty'])
        for entry in snapshot['asks']:
            self.update_entry(self.asks, entry['price'], entry['qty'])
        self.is_synced = True
        print("Snapshot applied.")

    def apply_update(self, update):
        for entry in update.get('bids', []):
            self.update_entry(self.bids, entry['price'], entry['qty'])
        for entry in update.get('asks', []):
            self.update_entry(self.asks, entry['price'], entry['qty'])
        
        if 'checksum' in update:
            server_checksum = int(update['checksum'])
            if not self.validate_checksum(server_checksum):
                print(f"CHECKSUM FAILED! Server: {update['checksum']}")
                self.is_synced = False
                raise ChecksumError("Checksum failed")
        # print("Update applied.")

    def update_entry(self, book, price_str, qty_str):
        # price_str and qty_str are Strings (from json.loads)
        price_dec = Decimal(price_str)
        qty_dec = Decimal(qty_str)
        
        key = -price_dec if book is self.bids else price_dec
        
        if qty_dec == 0:
            if key in book: del book[key]
        else:
            book[key] = (qty_dec, price_str, qty_str)

        if len(book) > self.depth:
                book.popitem()

    def validate_checksum(self, checksum):
        payload = ""
        
        # Asks (Low to High)
        for key in list(self.asks.keys())[:10]:
            qty_dec, price_str, qty_str = self.asks[key]
            payload += self._fmt(price_str) + self._fmt(qty_str)
            
        # Bids (High to Low)
        for key in list(self.bids.keys())[:10]:
            qty_dec, price_str, qty_str = self.bids[key]
            payload += self._fmt(price_str) + self._fmt(qty_str)
            
        return (zlib.crc32(payload.encode('utf-8')) & 0xffffffff) == checksum
    
    def _fmt(self, s):
        # Implements the logic from your screenshot exactly
        # s is a string like "45285.2" or "0.00100000"
        s = s.replace(".", "").lstrip("0")
        return s

async def run_kraken_feed():
    uri = "wss://ws.kraken.com/v2"
    symbol = "BTC/USD"
    recorder = DataRecorder(symbol)
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected...")
                await websocket.send(json.dumps({
                    "method": "subscribe",
                    "params": {"channel": "book", "symbol": [symbol], "depth": 25, "snapshot": True}
                }))
                
                manager = OrderBookManager(symbol, 25, recorder)
                
                while True:
                    response = await websocket.recv()
                    # CRITICAL: Parse numbers as Strings to preserve formatting
                    data = json.loads(response, parse_float=str, parse_int=str)
                    
                    if data.get("channel") == "book":
                        manager.handle_message(data)
                    elif data.get("channel") == "heartbeat":
                        continue
                        
        except (ChecksumError, websockets.ConnectionClosed, Exception) as e:
            print(f"Error: {e}. Reconnecting...")
            if recorder.buffer:
                await recorder.flush_buffer()
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(run_kraken_feed())
    except KeyboardInterrupt:
        print("Bot stopped.")