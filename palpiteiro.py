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
                    "Sua tarefa: analisar um repositório e dar UMA sugestão de melhoria. "
                    "Requisitos da sugestão: específica, prática, implementável em poucas horas. "
                    "Áreas: qualidade de código, documentação, testes, CI/CD, segurança, performance. "
                    "Idioma: português brasileiro. "
                    "Responda SOMENTE com a sugestão final, sem raciocínio ou análise. "
                    "Exemplo de resposta ideal:\n"
                    "**Adicione tratamento de erros nas chamadas HTTP**\n\n"
                    "As requisições em main.py não tratam exceções de rede, o que causa crashes silenciosos. "
                    "Envolva as chamadas httpx.get() em try/except e adicione retry com backoff exponencial "
                    "usando a biblioteca tenacity. Isso melhora a resiliência em ambientes instáveis."
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
        raw = response.model_dump()
        extra = raw.get("choices", [{}])[0].get("message", {})
        content = extra.get("reasoning_content", "") or extra.get("content", "")

    if not content:
        print(f"Resposta vazia do modelo. Raw: {response.model_dump_json()}")
        return ""

    # Modelos "thinking" despejam raciocinio no content.
    # Estrategia: encontrar o ultimo bloco "**titulo**" seguido de texto limpo.
    parts = re.split(r"(\*\*[^*]+\*\*)", content)

    suggestion = ""
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("**") and parts[i].endswith("**"):
            title = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ""
            body = body.strip().lstrip(":").strip()

            # Descartar se corpo comeca com padrao de raciocinio
            if re.match(r"^(\.|No entanto|Mas |Porém|Contudo|Entretanto|Vamos)", body):
                continue

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
            clean = []
            for s in sentences:
                if re.match(
                    r"^(Preciso|Vou |São \d|O usuário|Verificando|Rascunho|"
                    r"Isso tem|Isso é|Checando|Analis|Pontos de|A sugestão|Outra opção|"
                    r"Outra:|Formato|Texto:|Título:|Está bom|- )",
                    s,
                ):
                    break
                clean.append(s)

            if len(clean) >= 2:
                suggestion = f"{title}\n\n{' '.join(clean[:4])}"
                break

    return suggestion if suggestion else content.strip()[:500]


def _is_valid_suggestion(text: str) -> bool:
    invalid_patterns = ["título curto", "2-3 frases", "exemplo de resposta", "formato:"]
    lower = text.lower()
    return len(text) > 50 and not any(p in lower for p in invalid_patterns)


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

    if not _is_valid_suggestion(suggestion):
        print("Sugestão inválida, tentando novamente...")
        suggestion = generate_suggestion(details)

    if not _is_valid_suggestion(suggestion):
        print("Não foi possível gerar uma sugestão válida.")
        return

    message = f"*Palpiteiro* - Sugestão para `{details['name']}`\n\n{suggestion}\n\n_Modelo: {LLM_MODEL}_"
    print(f"\n{message}\n")

    send_telegram_message(message)


if __name__ == "__main__":
    main()
