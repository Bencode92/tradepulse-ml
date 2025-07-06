#!/usr/bin/env python3
"""
Script de diagnostic pour identifier les problèmes de formatage
==============================================================

Usage: python scripts/check_formatting.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list) -> tuple[int, str, str]:
    """Exécute une commande et retourne le code de retour + stdout + stderr"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def check_black_formatting():
    """Vérifie le formatage Black"""
    print("🔍 Vérification formatage Black...")

    code, stdout, stderr = run_command(
        [
            "python",
            "-m",
            "black",
            "--check",
            "--line-length=88",
            "scripts/",
        ]
    )

    if code == 0:
        print("✅ Black check: OK")
        return True
    else:
        print("❌ Black check: ÉCHEC")
        if stdout:
            print("📝 Sortie:")
            print(stdout)
        if stderr:
            print("🚨 Erreurs:")
            print(stderr)
        return False


def check_isort_formatting():
    """Vérifie le tri des imports"""
    print("\n🔍 Vérification tri imports isort...")

    code, stdout, stderr = run_command(
        [
            "python",
            "-m",
            "isort",
            "--check-only",
            "--profile=black",
            "scripts/",
        ]
    )

    if code == 0:
        print("✅ isort check: OK")
        return True
    else:
        print("❌ isort check: ÉCHEC")
        if stdout:
            print("📝 Sortie:")
            print(stdout)
        if stderr:
            print("🚨 Erreurs:")
            print(stderr)
        return False


def check_ruff_linting():
    """Vérifie le linting Ruff"""
    print("\n🔍 Vérification linting Ruff...")

    code, stdout, stderr = run_command(
        ["python", "-m", "ruff", "check", "scripts/"]
    )

    if code == 0:
        print("✅ Ruff check: OK")
        return True
    else:
        print("❌ Ruff check: ÉCHEC")
        if stdout:
            print("📝 Sortie:")
            print(stdout)
        if stderr:
            print("🚨 Erreurs:")
            print(stderr)
        return False


def list_python_files():
    """Liste tous les fichiers Python dans scripts/"""
    print("\n📁 Fichiers Python détectés:")
    scripts_dir = Path(__file__).parent

    for py_file in scripts_dir.glob("*.py"):
        if py_file.name != "check_formatting.py":  # Ignorer ce script
            print(f"  📄 {py_file.name}")


def main():
    print("🧪 DIAGNOSTIC FORMATAGE TRADEPULSE ML")
    print("=" * 50)

    list_python_files()

    # Vérifications
    black_ok = check_black_formatting()
    isort_ok = check_isort_formatting()
    ruff_ok = check_ruff_linting()

    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ:")
    print(f"  Black:  {'✅' if black_ok else '❌'}")
    print(f"  isort:  {'✅' if isort_ok else '❌'}")
    print(f"  Ruff:   {'✅' if ruff_ok else '❌'}")

    if all([black_ok, isort_ok, ruff_ok]):
        print("\n🎉 Tous les checks sont OK !")
        print("💡 Le problème de CI devrait être résolu")
        return 0
    else:
        print("\n🚨 Des problèmes de formatage persistent")
        print("💡 Commandes pour corriger :")
        if not black_ok:
            print("  python -m black --line-length=88 scripts/")
        if not isort_ok:
            print("  python -m isort --profile=black scripts/")
        if not ruff_ok:
            print("  python -m ruff check scripts/ --fix")
        return 1


if __name__ == "__main__":
    sys.exit(main())
