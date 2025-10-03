#!/usr/bin/env python3
"""
WebSocket Test Client für Live-Updates
====================================

Test-Client für die WebSocket-Funktionalität des Web-Interfaces.
"""

import asyncio
import socketio


async def test_websocket_connection():
    """Test WebSocket-Verbindung und Event-Handling"""

    print("🔌 WebSocket Test Client gestartet...")

    # Socket.IO Client erstellen
    sio = socketio.AsyncClient()

    @sio.event
    async def connect():
        print("✅ Mit Server verbunden!")

    @sio.event
    async def disconnect():
        print("❌ Verbindung getrennt")

    @sio.event
    async def simulation_started(data):
        print(f"🚀 Simulation gestartet: {data}")

    @sio.event
    async def progress(data):
        print(f"📊 Progress: {data['current']}/{data['total']} ({data['percentage']}%)")

    @sio.event
    async def simulation_finished(data):
        print(f"🏁 Simulation beendet: {data}")

    try:
        # Verbindung zum Server herstellen
        await sio.connect("http://localhost:5000")

        # Kurz warten für Events
        await asyncio.sleep(10)

        # Verbindung trennen
        await sio.disconnect()

    except Exception as e:
        print(f"❌ Fehler: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket_connection())
