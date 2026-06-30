import json
import os
import random
import re

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def get_repositories() -> list[dict]:
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    repos = []
    page = 1
    while True:
        response = httpx.get(
            f"https://api.github.com/users/{GITHUB_USERNAME}/repos",
            headers=headers,
            params={"per_page": 100, "page": page, "sort": "updated"},
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            break
        repos.extend(data)
        page += 1
    return repos


def get_repo_details(repo: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    full_name = repo["full_name"]

    readme_content = ""
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{full_name}/readme",
            headers=headers,
        )
        if resp.status_code == 200:
            import base64

            readme_content = base64.b64decode(resp.json()["content"]).decode("utf-8", errors="replace")
            readme_content = readme_content[:3000]
    except Exception:
        pass

    languages = {}
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{full_name}/languages",
            headers=headers,
        )
        if resp.status_code == 200:
            languages = resp.json()
    except Exception:
        pass

    recent_commits = []
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{full_name}/commits",
            headers=headers,
            params={"per_page": 5},
        )
        if resp.status_code == 200:
            recent_commits = [
                {"message": c["commit"]["message"], "date": c["commit"]["author"]["date"]}
                for c in resp.json()
            ]
    except Exception:
        pass

    open_issues = []
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{full_name}/issues",
            headers=headers,
            params={"per_page": 5, "state": "open"},
        )
        if resp.status_code == 200:
            open_issues = [{"title": i["title"], "labels": [l["name"] for l in i["labels"]]} for i in resp.json()]
    except Exception:
        pass

    return {
        "name": repo["name"],
        "full_name": full_name,
        "description": repo.get("description", ""),
        "language": repo.get("language", ""),
        "languages": languages,
        "stars": repo.get("stargazers_count", 0),
        "open_issues_count": repo.get("open_issues_count", 0),
        "last_push": repo.get("pushed_at", ""),
        "readme": readme_content,
        "recent_commits": recent_commits,
        "open_issues": open_issues,
        "has_wiki": repo.get("has_wiki", False),
        "license": repo.get("license"),
        "topics": repo.get("topics", []),
    }


def generate_suggestion(repo_details: dict) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    context = json.dumps(repo_details, ensure_ascii=False, indent=2, default=str)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um consultor de software experiente. "
                    "Analise o repositório abaixo e forneça UMA sugestão concreta e acionável de melhoria. "
                    "A sugestão deve ser específica, prática e implementável em poucas horas. "
                    "Considere: qualidade de código, documentação, testes, CI/CD, segurança, performance, "
                    "boas práticas da linguagem/framework. "
                    "Responda em português brasileiro, de forma direta e objetiva. "
                    "NÃO inclua seu raciocínio, análise intermediária ou passos de pensamento. "
                    "Responda APENAS com o resultado final no formato: "
                    "um título curto em negrito + 2-3 frases explicando o porquê e como implementar."
                ),
            },
            {
                "role": "user",
                "content": f"Repositório:\n{context}",
            },
        ],
        max_tokens=1000,
        temperature=0.7,
    )

    choice = response.choices[0]
    content = choice.message.content or ""

    if not content and hasattr(choice.message, "reasoning_content"):
        content = choice.message.reasoning_content or ""

    if not content:
        print(f"Resposta vazia do modelo. Raw: {response.model_dump_json()}")
        return ""

    # Modelos "thinking" despejam raciocinio antes da resposta final.
    # Extrair o ultimo bloco que comeca com titulo em negrito (**...**) + paragrafo.
    blocks = re.findall(
        r"(\*\*[^*]+\*\*[:\s]*\n+(?:[^*\n].+\n?)+)",
        content,
    )
    if blocks:
        content = blocks[-1].strip()

    return content


def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID não configurados.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}

    response = httpx.post(url, json=payload)
    if response.status_code == 200:
        print("Mensagem enviada com sucesso no Telegram.")
    else:
        # Fallback sem parse_mode caso o Markdown seja invalido
        print(f"Erro com Markdown ({response.status_code}), reenviando sem formatação...")
        payload.pop("parse_mode")
        response = httpx.post(url, json=payload)
        if response.status_code == 200:
            print("Mensagem enviada com sucesso (sem formatação).")
        else:
            print(f"Erro ao enviar no Telegram: {response.status_code} {response.text}")


def main():
    print("Buscando repositórios...")
    repos = get_repositories()

    if not repos:
        print("Nenhum repositório encontrado.")
        return

    repo = random.choice(repos)
    print(f"Repositório selecionado: {repo['full_name']}")

    print("Coletando detalhes...")
    details = get_repo_details(repo)

    print("Gerando sugestão com LLM...")
    suggestion = generate_suggestion(details)

    message = f"*Palpiteiro* - Sugestão para `{details['name']}`\n\n{suggestion}\n\n_Modelo: {LLM_MODEL}_"
    print(f"\n{message}\n")

    send_telegram_message(message)


if __name__ == "__main__":
    main()
