import subprocess

subprocess.run([
    "gguf-new-metadata",
    "--chat-template-file", "chat_template.jinja",
    "gemma-4-E4B-it.gguf",
    "gemma-4-E4B-it-fixed.gguf"
])