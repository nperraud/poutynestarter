from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).parents[0] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path.resolve(), override=True, verbose=True)
    
class LazyEnv:
    """Lazy environment variable."""

    def __init__(
        self,
        env_var,
        default=None,
        return_type=str,
        after_eval=None,
    ):
        """Construct lazy evaluated environment variable."""
        self.env_var = env_var
        self.default = default
        self.return_type = return_type
        self.after_eval = after_eval

    def eval(self):
        """Evaluate environment variable."""
        value = self.return_type(os.environ.get(self.env_var, self.default))

        if self.after_eval:
            self.after_eval(value)

        return value
    
PATH_ROOT = str(Path(__file__).parents[0])
PATH_DATA = LazyEnv("PATH_DATA", Path(PATH_ROOT) / Path("data")).eval()

PATH_SUMMARY = LazyEnv("PATH_SUMMARY", Path(PATH_DATA) / Path("summary")).eval()
PATH_CHECKPOINTS = LazyEnv("PATH_CHECKPOINTS", Path(PATH_DATA) / Path("checkpoints")).eval()

Path(PATH_SUMMARY).mkdir(parents=True, exist_ok=True)
Path(PATH_CHECKPOINTS).mkdir(parents=True, exist_ok=True)

