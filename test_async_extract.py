import json

def parse_json_array(raw: str) -> list:
    json_str = raw.strip()
    if "```" in json_str:
        parts = json_str.split("```")
        if len(parts) >= 3:
            json_str = parts[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()
    try:
        topics = json.loads(json_str)
        if isinstance(topics, list):
            return topics
        elif isinstance(topics, dict) and "topics" in topics:
            return topics["topics"]
    except json.JSONDecodeError:
        pass
    return []
