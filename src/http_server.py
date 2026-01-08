#!/usr/bin/env python3
"""
SoundCloud MCP Server
Downloads tracks from SoundCloud and plays them on Chromecast via Home Assistant
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import requests
from flask import Flask, Response, jsonify, request, send_from_directory
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
PORT = int(os.environ.get('PORT', 3040))
DATA_DIR = os.environ.get('DATA_DIR', '/data')
BASE_URL = os.environ.get('BASE_URL', 'https://mcp-soundcloud.local.jbmurphy.com')
HA_URL = os.environ.get('HA_URL', 'https://homeassistant.local.jbmurphy.com:8123')
HA_TOKEN = os.environ.get('HA_TOKEN', '')

app = Flask(__name__)

# MCP Server instance
mcp_server = Server("soundcloud-mcp-server")


class SoundCloudManager:
    """Manages SoundCloud downloads and audio library"""

    def __init__(self, data_dir: str, base_url: str):
        self.data_dir = Path(data_dir)
        self.base_url = base_url.rstrip('/')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.subscriptions_file = self.data_dir / 'podcasts.json'
        logger.info(f"SoundCloud manager initialized: data_dir={data_dir}, base_url={base_url}")

    def load_subscriptions(self) -> dict[str, Any]:
        """Load subscriptions from podcasts.json"""
        if self.subscriptions_file.exists():
            try:
                with open(self.subscriptions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading subscriptions: {e}")
                return {"subscriptions": []}
        return {"subscriptions": []}

    def save_subscriptions(self, data: dict[str, Any]) -> bool:
        """Save subscriptions to podcasts.json"""
        try:
            with open(self.subscriptions_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving subscriptions: {e}")
            return False

    def list_subscriptions(self) -> dict[str, Any]:
        """List all subscriptions"""
        data = self.load_subscriptions()
        return {
            "success": True,
            "subscriptions": data.get("subscriptions", []),
            "count": len(data.get("subscriptions", []))
        }

    def add_subscription(self, url: str, name: Optional[str] = None) -> dict[str, Any]:
        """Add a SoundCloud user/playlist to subscriptions"""
        try:
            # Extract name from URL if not provided
            if not name:
                parts = url.rstrip('/').split('/')
                if 'soundcloud.com' in url and len(parts) > 3:
                    name = parts[3]
                else:
                    name = 'unknown'

            clean_name = self.clean_name(name)
            data = self.load_subscriptions()

            # Check if already subscribed
            for sub in data.get("subscriptions", []):
                if sub.get("url") == url or sub.get("name") == clean_name:
                    return {
                        "success": False,
                        "error": f"Already subscribed to '{name}'"
                    }

            # Add new subscription
            new_sub = {
                "name": clean_name,
                "url": url,
                "added": datetime.now().isoformat(),
                "last_sync": None
            }
            data.setdefault("subscriptions", []).append(new_sub)

            if self.save_subscriptions(data):
                return {
                    "success": True,
                    "subscription": new_sub,
                    "message": f"Added subscription for '{clean_name}'"
                }
            return {"success": False, "error": "Failed to save subscription"}

        except Exception as e:
            logger.error(f"Error adding subscription: {e}")
            return {"success": False, "error": str(e)}

    def remove_subscription(self, name: str) -> dict[str, Any]:
        """Remove a subscription by name"""
        try:
            clean_name = self.clean_name(name)
            data = self.load_subscriptions()

            original_count = len(data.get("subscriptions", []))
            data["subscriptions"] = [
                s for s in data.get("subscriptions", [])
                if s.get("name") != clean_name and s.get("name") != name
            ]

            if len(data["subscriptions"]) == original_count:
                return {
                    "success": False,
                    "error": f"Subscription '{name}' not found"
                }

            if self.save_subscriptions(data):
                return {
                    "success": True,
                    "message": f"Removed subscription for '{name}'"
                }
            return {"success": False, "error": "Failed to save changes"}

        except Exception as e:
            logger.error(f"Error removing subscription: {e}")
            return {"success": False, "error": str(e)}

    def sync_subscriptions(self, limit: Optional[int] = None) -> dict[str, Any]:
        """Sync all subscriptions - download new tracks"""
        try:
            data = self.load_subscriptions()
            subscriptions = data.get("subscriptions", [])

            if not subscriptions:
                return {
                    "success": True,
                    "message": "No subscriptions to sync",
                    "results": []
                }

            results = []
            for sub in subscriptions:
                url = sub.get("url")
                name = sub.get("name")

                logger.info(f"Syncing subscription: {name}" + (f" (limit: {limit})" if limit else ""))
                result = self.download_tracks(url, name, limit)

                results.append({
                    "name": name,
                    "success": result.get("success", False),
                    "track_count": result.get("track_count", 0),
                    "error": result.get("error")
                })

                # Update last_sync timestamp
                sub["last_sync"] = datetime.now().isoformat()

            # Save updated timestamps
            self.save_subscriptions(data)

            total_tracks = sum(r.get("track_count", 0) for r in results)
            successful = sum(1 for r in results if r.get("success"))

            return {
                "success": True,
                "message": f"Synced {successful}/{len(subscriptions)} subscriptions",
                "total_tracks": total_tracks,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error syncing subscriptions: {e}")
            return {"success": False, "error": str(e)}

    def clean_name(self, name: str) -> str:
        """Clean a name for use as directory/file name"""
        cleaned = re.sub(r'[^\w\s-]', '', name)
        cleaned = re.sub(r'[-\s]+', '_', cleaned)
        return cleaned.strip('_').lower()

    def download_tracks(self, url: str, artist_name: Optional[str] = None, limit: Optional[int] = None) -> dict[str, Any]:
        """Download tracks from a SoundCloud URL"""
        try:
            # Extract artist name from URL if not provided
            if not artist_name:
                # URL format: https://soundcloud.com/username/tracks
                parts = url.rstrip('/').split('/')
                if 'soundcloud.com' in url:
                    artist_name = parts[3] if len(parts) > 3 else 'unknown'
                else:
                    artist_name = 'unknown'

            artist_name = self.clean_name(artist_name)
            artist_dir = self.data_dir / artist_name
            artist_dir.mkdir(parents=True, exist_ok=True)
            archive_file = artist_dir / f'{artist_name}_archive.txt'

            # Strip /tracks from URL since we use -t flag
            clean_url = url.rstrip('/').removesuffix('/tracks')
            logger.info(f"Downloading from {clean_url} to {artist_dir}" + (f" (limit: {limit})" if limit else ""))

            # Run scdl to download tracks
            cmd = [
                'scdl',
                '-l', clean_url,
                '--path', str(artist_dir),
                '--download-archive', str(archive_file),
                '-t'  # Download tracks
            ]

            # Pass limit to yt-dlp via --yt-dlp-args (scdl v3 wraps yt-dlp)
            if limit and limit > 0:
                cmd.extend(['--yt-dlp-args', f'--playlist-end {limit}'])

            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            # Log output regardless of return code
            logger.info(f"scdl stdout: {result.stdout}")
            logger.info(f"scdl stderr: {result.stderr}")
            logger.info(f"scdl return code: {result.returncode}")

            if result.returncode != 0:
                logger.error(f"scdl error: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr or result.stdout or "Download failed",
                    "artist": artist_name,
                    "output": result.stdout
                }

            # Count downloaded files
            tracks = list(artist_dir.glob('*.mp3')) + list(artist_dir.glob('*.m4a'))

            return {
                "success": True,
                "artist": artist_name,
                "track_count": len(tracks),
                "directory": str(artist_dir),
                "output": result.stdout
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Download timed out after 10 minutes"}
        except Exception as e:
            logger.error(f"Download error: {e}")
            return {"success": False, "error": str(e)}

    def list_artists(self) -> dict[str, Any]:
        """List all downloaded artists"""
        try:
            artists = []
            for artist_dir in self.data_dir.iterdir():
                if artist_dir.is_dir():
                    tracks = list(artist_dir.glob('*.mp3')) + list(artist_dir.glob('*.m4a'))
                    if tracks:
                        artists.append({
                            "name": artist_dir.name,
                            "track_count": len(tracks),
                            "path": str(artist_dir)
                        })

            return {
                "success": True,
                "artists": sorted(artists, key=lambda x: x['name']),
                "count": len(artists)
            }
        except Exception as e:
            logger.error(f"Error listing artists: {e}")
            return {"success": False, "error": str(e)}

    def list_tracks(self, artist: str) -> dict[str, Any]:
        """List tracks for an artist"""
        try:
            artist_dir = self.data_dir / self.clean_name(artist)
            if not artist_dir.exists():
                return {"success": False, "error": f"Artist '{artist}' not found"}

            tracks = []
            for track_path in sorted(artist_dir.glob('*.mp3')) + sorted(artist_dir.glob('*.m4a')):
                filename = track_path.name
                encoded_filename = quote(filename)
                tracks.append({
                    "title": track_path.stem,
                    "filename": filename,
                    "url": f"{self.base_url}/audio/{artist}/{encoded_filename}",
                    "size_mb": round(track_path.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(track_path.stat().st_mtime).isoformat()
                })

            return {
                "success": True,
                "artist": artist,
                "tracks": tracks,
                "count": len(tracks)
            }
        except Exception as e:
            logger.error(f"Error listing tracks: {e}")
            return {"success": False, "error": str(e)}

    def get_track_url(self, artist: str, track_filename: str) -> Optional[str]:
        """Get the full URL for a track"""
        artist_dir = self.data_dir / self.clean_name(artist)
        track_path = artist_dir / track_filename
        if track_path.exists():
            encoded_filename = quote(track_filename)
            return f"{self.base_url}/audio/{artist}/{encoded_filename}"
        return None

    def search_tracks(self, query: str) -> dict[str, Any]:
        """Search for tracks across all artists"""
        try:
            query_lower = query.lower()
            results = []

            for artist_dir in self.data_dir.iterdir():
                if artist_dir.is_dir():
                    for track_path in list(artist_dir.glob('*.mp3')) + list(artist_dir.glob('*.m4a')):
                        if query_lower in track_path.stem.lower():
                            filename = track_path.name
                            encoded_filename = quote(filename)
                            results.append({
                                "artist": artist_dir.name,
                                "title": track_path.stem,
                                "filename": filename,
                                "url": f"{self.base_url}/audio/{artist_dir.name}/{encoded_filename}"
                            })

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {"success": False, "error": str(e)}


class HomeAssistantPlayer:
    """Plays audio on Chromecast via Home Assistant"""

    def __init__(self, ha_url: str, ha_token: str):
        self.ha_url = ha_url.rstrip('/')
        self.token = ha_token
        self.headers = {
            'Authorization': f'Bearer {ha_token}',
            'Content-Type': 'application/json'
        }

    def list_media_players(self) -> dict[str, Any]:
        """List available media players from Home Assistant"""
        try:
            url = f"{self.ha_url}/api/states"
            response = requests.get(url, headers=self.headers, verify=False, timeout=10)
            response.raise_for_status()

            entities = response.json()
            players = []
            for entity in entities:
                if entity['entity_id'].startswith('media_player.'):
                    players.append({
                        "entity_id": entity['entity_id'],
                        "friendly_name": entity['attributes'].get('friendly_name', entity['entity_id']),
                        "state": entity['state']
                    })

            return {
                "success": True,
                "players": sorted(players, key=lambda x: x['friendly_name']),
                "count": len(players)
            }
        except Exception as e:
            logger.error(f"Error listing media players: {e}")
            return {"success": False, "error": str(e)}

    def play_media(self, entity_id: str, media_url: str, media_type: str = "audio/mpeg") -> dict[str, Any]:
        """Play media on a Home Assistant media player"""
        try:
            url = f"{self.ha_url}/api/services/media_player/play_media"
            data = {
                "entity_id": entity_id,
                "media_content_id": media_url,
                "media_content_type": media_type
            }

            logger.info(f"Playing {media_url} on {entity_id}")
            response = requests.post(url, headers=self.headers, json=data, verify=False, timeout=10)
            response.raise_for_status()

            return {
                "success": True,
                "entity_id": entity_id,
                "media_url": media_url,
                "status": "Playing"
            }
        except Exception as e:
            logger.error(f"Error playing media: {e}")
            return {"success": False, "error": str(e)}


# Initialize managers
sc_manager = SoundCloudManager(DATA_DIR, BASE_URL)
ha_player = HomeAssistantPlayer(HA_URL, HA_TOKEN)


# Define MCP Tools
@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available SoundCloud tools"""
    return [
        Tool(
            name="download",
            description="Download tracks from a SoundCloud URL (artist page or playlist)",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "SoundCloud URL (e.g., https://soundcloud.com/artist/tracks)"
                    },
                    "artist_name": {
                        "type": "string",
                        "description": "Optional custom name for the artist folder"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tracks to download (optional, downloads all if not specified)"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="list_artists",
            description="List all downloaded artists and their track counts",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_tracks",
            description="List all tracks for a specific artist",
            inputSchema={
                "type": "object",
                "properties": {
                    "artist": {
                        "type": "string",
                        "description": "Artist name (folder name)"
                    }
                },
                "required": ["artist"]
            }
        ),
        Tool(
            name="search",
            description="Search for tracks by name across all artists",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="play",
            description="Play a track on a Chromecast/speaker via Home Assistant",
            inputSchema={
                "type": "object",
                "properties": {
                    "artist": {
                        "type": "string",
                        "description": "Artist name"
                    },
                    "track": {
                        "type": "string",
                        "description": "Track filename or search term"
                    },
                    "player": {
                        "type": "string",
                        "description": "Media player entity_id (e.g., media_player.family_room_speaker)"
                    }
                },
                "required": ["artist", "track", "player"]
            }
        ),
        Tool(
            name="play_url",
            description="Play any audio URL on a Chromecast/speaker via Home Assistant",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Audio URL to play"
                    },
                    "player": {
                        "type": "string",
                        "description": "Media player entity_id (e.g., media_player.family_room_speaker)"
                    }
                },
                "required": ["url", "player"]
            }
        ),
        Tool(
            name="list_players",
            description="List available media players (Chromecasts, speakers) from Home Assistant",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_subscriptions",
            description="List all SoundCloud subscriptions being tracked",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="add_subscription",
            description="Add a SoundCloud artist/playlist to track for new downloads",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "SoundCloud URL (e.g., https://soundcloud.com/aboveandbeyond/tracks)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional custom name for the subscription"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="remove_subscription",
            description="Remove a SoundCloud subscription",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the subscription to remove"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="sync",
            description="Sync all subscriptions - download new tracks from all tracked artists",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tracks to download per artist (optional, downloads all if not specified)"
                    }
                },
                "required": []
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute SoundCloud tool"""
    try:
        logger.info(f"Executing tool: {name} with arguments: {arguments}")

        if name == "download":
            url = arguments.get("url")
            artist_name = arguments.get("artist_name")
            limit = arguments.get("limit")

            if not url:
                raise ValueError("url parameter is required")

            result = sc_manager.download_tracks(url, artist_name, limit)

            if result.get("success"):
                response_text = f"✓ Downloaded tracks for {result['artist']}\n"
                response_text += f"Track count: {result['track_count']}\n"
                response_text += f"Directory: {result['directory']}"
            else:
                response_text = f"✗ Download failed: {result.get('error')}"

        elif name == "list_artists":
            result = sc_manager.list_artists()

            if result.get("success"):
                response_text = f"✓ Found {result['count']} artists:\n\n"
                for artist in result['artists']:
                    response_text += f"• {artist['name']} ({artist['track_count']} tracks)\n"
            else:
                response_text = f"✗ Error: {result.get('error')}"

        elif name == "list_tracks":
            artist = arguments.get("artist")
            if not artist:
                raise ValueError("artist parameter is required")

            result = sc_manager.list_tracks(artist)

            if result.get("success"):
                response_text = f"✓ {result['count']} tracks for {result['artist']}:\n\n"
                for track in result['tracks']:
                    response_text += f"• {track['title']} ({track['size_mb']} MB)\n"
                    response_text += f"  URL: {track['url']}\n"
            else:
                response_text = f"✗ Error: {result.get('error')}"

        elif name == "search":
            query = arguments.get("query")
            if not query:
                raise ValueError("query parameter is required")

            result = sc_manager.search_tracks(query)

            if result.get("success"):
                response_text = f"✓ Found {result['count']} tracks matching '{query}':\n\n"
                for track in result['results'][:20]:  # Limit to 20 results
                    response_text += f"• {track['artist']} - {track['title']}\n"
            else:
                response_text = f"✗ Error: {result.get('error')}"

        elif name == "play":
            artist = arguments.get("artist")
            track = arguments.get("track")
            player = arguments.get("player")

            if not all([artist, track, player]):
                raise ValueError("artist, track, and player parameters are required")

            # Find the track
            tracks_result = sc_manager.list_tracks(artist)
            if not tracks_result.get("success"):
                response_text = f"✗ Error: {tracks_result.get('error')}"
            else:
                # Find matching track
                matching_track = None
                track_lower = track.lower()
                for t in tracks_result['tracks']:
                    if track_lower in t['filename'].lower() or track_lower in t['title'].lower():
                        matching_track = t
                        break

                if not matching_track:
                    response_text = f"✗ Track '{track}' not found for artist '{artist}'"
                else:
                    # Determine media type
                    media_type = "audio/mp4" if matching_track['filename'].endswith('.m4a') else "audio/mpeg"

                    # Play via Home Assistant
                    play_result = ha_player.play_media(player, matching_track['url'], media_type)

                    if play_result.get("success"):
                        response_text = f"✓ Playing: {matching_track['title']}\n"
                        response_text += f"On: {player}\n"
                        response_text += f"URL: {matching_track['url']}"
                    else:
                        response_text = f"✗ Playback failed: {play_result.get('error')}"

        elif name == "play_url":
            url = arguments.get("url")
            player = arguments.get("player")

            if not all([url, player]):
                raise ValueError("url and player parameters are required")

            # Determine media type from URL
            if url.endswith('.m4a'):
                media_type = "audio/mp4"
            elif url.endswith('.mp3'):
                media_type = "audio/mpeg"
            else:
                media_type = "audio/mpeg"  # Default

            result = ha_player.play_media(player, url, media_type)

            if result.get("success"):
                response_text = f"✓ Playing URL on {player}\n"
                response_text += f"URL: {url}"
            else:
                response_text = f"✗ Playback failed: {result.get('error')}"

        elif name == "list_players":
            result = ha_player.list_media_players()

            if result.get("success"):
                response_text = f"✓ Found {result['count']} media players:\n\n"
                for player in result['players']:
                    state_icon = "🟢" if player['state'] not in ['off', 'unavailable'] else "⚪"
                    response_text += f"{state_icon} {player['friendly_name']}\n"
                    response_text += f"   {player['entity_id']} ({player['state']})\n"
            else:
                response_text = f"✗ Error: {result.get('error')}"

        elif name == "list_subscriptions":
            result = sc_manager.list_subscriptions()

            if result.get("success"):
                if result['count'] == 0:
                    response_text = "No subscriptions configured.\n\nUse add_subscription to add SoundCloud artists to track."
                else:
                    response_text = f"✓ {result['count']} subscriptions:\n\n"
                    for sub in result['subscriptions']:
                        last_sync = sub.get('last_sync', 'Never')
                        if last_sync and last_sync != 'Never':
                            last_sync = last_sync[:19].replace('T', ' ')
                        response_text += f"• {sub['name']}\n"
                        response_text += f"  URL: {sub['url']}\n"
                        response_text += f"  Last sync: {last_sync}\n\n"
            else:
                response_text = f"✗ Error: {result.get('error')}"

        elif name == "add_subscription":
            url = arguments.get("url")
            sub_name = arguments.get("name")

            if not url:
                raise ValueError("url parameter is required")

            result = sc_manager.add_subscription(url, sub_name)

            if result.get("success"):
                response_text = f"✓ {result['message']}\n\n"
                response_text += f"URL: {result['subscription']['url']}\n"
                response_text += "Run 'sync' to download tracks."
            else:
                response_text = f"✗ Error: {result.get('error')}"

        elif name == "remove_subscription":
            sub_name = arguments.get("name")

            if not sub_name:
                raise ValueError("name parameter is required")

            result = sc_manager.remove_subscription(sub_name)

            if result.get("success"):
                response_text = f"✓ {result['message']}"
            else:
                response_text = f"✗ Error: {result.get('error')}"

        elif name == "sync":
            limit = arguments.get("limit")
            result = sc_manager.sync_subscriptions(limit)

            if result.get("success"):
                response_text = f"✓ {result['message']}\n"
                response_text += f"Total tracks: {result.get('total_tracks', 0)}\n\n"
                for r in result.get('results', []):
                    status = "✓" if r.get('success') else "✗"
                    response_text += f"{status} {r['name']}: {r.get('track_count', 0)} tracks"
                    if r.get('error'):
                        response_text += f" (Error: {r['error']})"
                    response_text += "\n"
            else:
                response_text = f"✗ Error: {result.get('error')}"

        else:
            raise ValueError(f"Unknown tool: {name}")

        return [TextContent(type="text", text=response_text)]

    except Exception as e:
        logger.error(f"Error in call_tool: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# Flask HTTP Endpoints
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "data_dir": DATA_DIR,
        "base_url": BASE_URL
    }), 200


@app.route('/audio/<artist>/<filename>')
def serve_audio(artist, filename):
    """Serve audio files"""
    try:
        artist_dir = sc_manager.data_dir / sc_manager.clean_name(artist)
        if not artist_dir.exists():
            return jsonify({"error": "Artist not found"}), 404

        # Determine content type
        if filename.endswith('.m4a'):
            mimetype = 'audio/mp4'
        else:
            mimetype = 'audio/mpeg'

        return send_from_directory(artist_dir, filename, mimetype=mimetype)
    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        return jsonify({"error": str(e)}), 404


@app.route('/mcp/list_tools', methods=['GET'])
def http_list_tools():
    """REST endpoint to list available tools"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tools = loop.run_until_complete(list_tools())
            return jsonify({
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    }
                    for tool in tools
                ]
            })
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error in list_tools: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/mcp/call_tool', methods=['POST'])
def http_call_tool():
    """REST endpoint to call a tool"""
    try:
        data = request.json
        tool_name = data.get('name')
        arguments = data.get('arguments', {})

        if not tool_name:
            return jsonify({"error": "Missing 'name' parameter"}), 400

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(call_tool(tool_name, arguments))
            return jsonify({
                "content": [
                    {
                        "type": content.type,
                        "text": content.text
                    }
                    for content in result
                ],
                "isError": False
            })
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error in call_tool: {e}")
        return jsonify({
            "content": [{"type": "text", "text": f"Error: {str(e)}"}],
            "isError": True
        }), 500


@app.route('/mcp', methods=['POST'])
def mcp_http():
    """HTTP transport for MCP (JSON-RPC 2.0)"""
    try:
        data = request.json
        method = data.get('method')
        params = data.get('params', {})
        msg_id = data.get('id')

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if method == 'initialize':
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "soundcloud-mcp-server",
                        "version": "1.0.0"
                    }
                }
            elif method == 'tools/list':
                tools = loop.run_until_complete(list_tools())
                result = {
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema
                        }
                        for tool in tools
                    ]
                }
            elif method == 'tools/call':
                tool_name = params.get('name')
                arguments = params.get('arguments', {})
                content = loop.run_until_complete(call_tool(tool_name, arguments))
                result = {
                    "content": [
                        {
                            "type": c.type,
                            "text": c.text
                        }
                        for c in content
                    ]
                }
            elif method == 'notifications/initialized':
                return '', 204
            else:
                return jsonify({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }), 404

            return jsonify({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result
            })
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error in MCP HTTP handler: {e}")
        return jsonify({
            "jsonrpc": "2.0",
            "id": msg_id if 'msg_id' in locals() else None,
            "error": {"code": -32603, "message": str(e)}
        }), 500


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--stdio':
        logger.info("Starting SoundCloud MCP server in stdio mode")
        asyncio.run(stdio_server(mcp_server))
    else:
        logger.info(f"Starting SoundCloud MCP HTTP server on port {PORT}")
        logger.info(f"Data directory: {DATA_DIR}")
        logger.info(f"Base URL: {BASE_URL}")
        logger.info(f"Home Assistant: {HA_URL}")
        app.run(host='0.0.0.0', port=PORT, debug=False)
