import asyncio

import websockets


async def callback_function(data):
    print(f"Received data: {data}")


async def websocket_client(uri, callback):
    async with websockets.connect(uri) as websocket:
        while True:
            await websocket.send("Hello, world!")
            data = await websocket.recv()
            await callback(data)


uri = "ws://localhost:8000/ws/some_path/"
asyncio.get_event_loop().run_until_complete(websocket_client(uri, callback_function))
