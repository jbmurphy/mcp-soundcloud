# CLAUDE.md

This file provides guidance to Claude Code when working with the SoundCloud MCP Server.

## Project Overview

MCP server for downloading tracks from SoundCloud and playing them on Chromecast/speakers via Home Assistant. Self-contained service that runs on nas01.

## Architecture

```
SoundCloud → scdl (download) → Local Storage → Flask (serve) → Home Assistant → Chromecast
```

**Language:** Python 3.11
**Framework:** Flask
**Transport:** HTTP (MCP protocol) on port 3040
**Deployment:** nas01 via Docker

## Tools

| Tool | Description |
|------|-------------|
| `download` | Download tracks from a SoundCloud URL |
| `list_artists` | List all downloaded artists |
| `list_tracks` | List tracks for an artist |
| `search` | Search tracks by name |
| `play` | Play a track on a speaker via Home Assistant |
| `play_url` | Play any audio URL on a speaker |
| `list_players` | List available media players from Home Assistant |
| `list_subscriptions` | List all SoundCloud subscriptions being tracked |
| `add_subscription` | Add a SoundCloud artist/playlist to track |
| `remove_subscription` | Remove a subscription |
| `sync` | Download new tracks from all subscriptions |

## Endpoints

- `GET /health` - Health check
- `GET /audio/{artist}/{filename}` - Serve audio files
- `GET /mcp/list_tools` - List available tools
- `POST /mcp/call_tool` - Call a tool
- `POST /mcp` - MCP JSON-RPC endpoint

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 3040 |
| `DATA_DIR` | Directory for downloaded tracks | /data |
| `BASE_URL` | Public URL for serving audio (use IP for Chromecast) | http://192.168.11.14:3040 |
| `HA_URL` | Home Assistant URL | https://nas01.local.jbmurphy.com:8123 |
| `HA_TOKEN` | Home Assistant long-lived access token | (required) |

**Note:** `BASE_URL` must use an IP address (not hostname) because Chromecasts hardcode Google DNS and cannot resolve local `.jbmurphy.com` domains.

## Building and Deploying

### Build locally and push to registry:

```bash
cd mcp-soundcloud

# Build for amd64 (nas01 architecture)
docker buildx build --platform linux/amd64 -t registry.local.jbmurphy.com/mcp-soundcloud:latest --push .
```

### Deploy on nas01:

```bash
# SSH to nas01 or use Portainer
docker pull registry.local.jbmurphy.com/mcp-soundcloud:latest

# Create .env file with HA_TOKEN
echo "HA_TOKEN=your_home_assistant_token" > .env

# Start the container
docker-compose up -d
```

### Setup DNS and Proxy:

1. Create DNS record: `mcp-soundcloud.local.jbmurphy.com` → proxy IP
2. Create proxy host forwarding to `nas01.local.jbmurphy.com:3040` with SSL

### Register with MCP Aggregator:

```bash
redis-cli -h redis.local.jbmurphy.com -p 32383 SET mcp:soundcloud https://mcp-soundcloud.local.jbmurphy.com
```

## Usage Examples

### Download from SoundCloud:
```python
from mcp import discover
discover.call_tool('soundcloud_download', url='https://soundcloud.com/aboveandbeyond/tracks')
```

### List artists:
```python
discover.call_tool('soundcloud_list_artists')
```

### Play a track:
```python
discover.call_tool('soundcloud_play',
    artist='aboveandbeyond',
    track='group therapy',
    player='media_player.family_room_speaker'
)
```

### List available speakers:
```python
discover.call_tool('soundcloud_list_players')
```

### Subscription Management:
```python
# Add a subscription
discover.call_tool('soundcloud_add_subscription', url='https://soundcloud.com/aboveandbeyond/tracks')

# List all subscriptions
discover.call_tool('soundcloud_list_subscriptions')

# Sync all subscriptions (download new tracks)
discover.call_tool('soundcloud_sync')

# Sync with limit (download only last N tracks per artist)
discover.call_tool('soundcloud_sync', limit=5)

# Remove a subscription
discover.call_tool('soundcloud_remove_subscription', name='aboveandbeyond')
```

## Migrating Existing Data

To migrate data from the old jbapi-rss setup:

```bash
# Copy existing podcasts to the new data volume
docker cp /path/to/old/podcasts/. mcp-soundcloud:/data/
```

## Dependencies

- **scdl** - SoundCloud downloader CLI
- **ffmpeg** - Audio processing
- **Flask** - HTTP server
- **mcp** - Model Context Protocol SDK
- **requests** - HTTP client for Home Assistant API

## Related Services

- **mcp-homeassistant** - Used for media player control
- **Home Assistant** - Chromecast/speaker integration
- **mcp-aggregator** - Tool discovery
