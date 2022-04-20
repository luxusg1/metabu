import hydra

from experiments.task1 import run

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg) -> None:
    run(cfg)

if __name__ == "__main__":
    my_app()
