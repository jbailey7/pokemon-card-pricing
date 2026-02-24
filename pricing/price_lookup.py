from urllib.parse import urlencode


def tcgplayer_url(card: dict) -> str:
    params = urlencode({
        "q": f"{card['name']} {card['set_name']}",
        "productLineName": "pokemon",
    })
    return f"https://www.tcgplayer.com/search/pokemon/product?{params}"
