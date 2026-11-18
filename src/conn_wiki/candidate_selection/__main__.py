"""Entry point: python -m candidate_judge."""
from .jev import main
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    main()