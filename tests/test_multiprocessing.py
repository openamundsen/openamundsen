from .conftest import base_config
import multiprocessing
import openamundsen as oa


def run_model(model):
    model.run()


def test_multiprocessing():
    config = base_config()
    config.end_date = '2020-01-16'
    config.output_data.timeseries.format = 'memory'
    config.output_data.grids.format = 'memory'

    config.snow.model = 'multilayer'
    model1 = oa.OpenAmundsen(config)
    model1.initialize()

    config.snow.model = 'cryolayers'
    model2 = oa.OpenAmundsen(config)
    model2.initialize()

    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool() as pool:
        pool.map(run_model, [model1, model2])
