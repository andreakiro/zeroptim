# expose factories at root level :)
from zeroptim.dataset._factory import DataLoaderFactory
from zeroptim.models._factory import ModelFactory
from zeroptim.optim._factory import OptimFactory

get_model = ModelFactory.get_model
get_loader = DataLoaderFactory.get_loader
get_optim = OptimFactory.get_optimizer
