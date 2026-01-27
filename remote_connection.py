import asyncio
import websockets
import json
# USE THIS FILE TO SEE WHAT THE LIVE ORDER BOOK DATA FROM KRAKEN LOOKS LIKE
async def view_raw_feed():
    # Kraken WebSocket V2 URL
    uri = "wss://ws.kraken.com/v2"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to Kraken V2 API...")
        
        # 1. Create Subscription Message
        # We use depth=10 so your screen isn't flooded with 100 lines of numbers
        subscribe_msg = {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": ["BTC/USD"],
                "depth": 10,  
                "snapshot": True
            }
        }
        
        # 2. Send the request
        await websocket.send(json.dumps(subscribe_msg))
        print("Subscription sent. Waiting for data...\n")
        
        # 3. Listen loop
        try:
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                # Check message type to print helpful labels
                msg_type = data.get("type")
                
                if msg_type == "snapshot":
                    print("--- [SNAPSHOT RECEIVED] (This is the full book) ---")
                    print(json.dumps(data, indent=2)) # indent=2 makes it pretty
                    print("\n" + "="*50 + "\n")
                    
                elif msg_type == "update":
                    print("--- [UPDATE RECEIVED] (Only changes) ---")
                    print(json.dumps(data, indent=2))
                    print("-" * 30)
                    
                else:
                    # Heartbeats or Status messages
                    print(f"[System Message]: {data}")

        except KeyboardInterrupt:
            print("\nStopping script...")

# Run the async loop
if __name__ == "__main__":
    asyncio.run(view_raw_feed())