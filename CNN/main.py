"""哈哈哈万能的main file"""
from utilities import get_interaction

from configuration import settings, initialize_environment, finalize_environment
from datasource import DataLoader
from architecture import NeuralModel
from losses import Criterion
from strategy import StrategyOptimizer
from execution import ExecutionManager

def process_task(worker_id, configuration):
    configuration.worker_id = worker_id
    configuration = initialize_environment(configuration)

    data_loaders = DataLoader(configuration).prepare_data()
    network = NeuralModel(configuration)
    network.distribute_computation()

    optimizer = StrategyOptimizer(configuration, network)

    loss_computation = Criterion(configuration, model=network, optimizer=optimizer)

    execution_plan = ExecutionManager(configuration, network, loss_computation, optimizer, data_loaders)

    if configuration.interactive_mode:
        get_interaction(local=locals())
        exit()

    if configuration.demo_mode:
        execution_plan.run_evaluation(configuration.initial_epoch, mode='demo')
        exit()

    for epoch in range(1, configuration.initial_epoch):
        if configuration.enable_validation:
            if epoch % configuration.validation_interval == 0:
                execution_plan.perform_evaluation(epoch, 'validation')

        if configuration.enable_testing:
            if epoch % configuration.testing_interval == 0:
                execution_plan.perform_evaluation(epoch, 'testing')

    for epoch in range(configuration.initial_epoch, configuration.final_epoch + 1):
        if configuration.enable_training:
            execution_plan.execute_training_cycle(epoch)

        if configuration.enable_validation:
            if epoch % configuration.validation_interval == 0:
                if execution_plan.current_epoch != epoch:
                    execution_plan.load_state(epoch)
                execution_plan.validate(epoch)

        if configuration.enable_testing:
            if epoch % configuration.testing_interval == 0:
                if execution_plan.current_epoch != epoch:
                    execution_plan.load_state(epoch)
                execution_plan.test(epoch)

        if worker_id == 0 or not configuration.is_distributed:
            print('Progressing...')

    execution_plan.save_images_in_background()

    finalize_environment(configuration)

def run():
    process_task(settings.worker_id, settings)

if __name__ == "__main__":
    run()