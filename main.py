import hydra

from experiments.tasks import run_task1

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg) -> None:
    run_task1(cfg)

if __name__ == "__main__":
    my_app()
