# Configuration Management
class Config:
    def __init__(self) -> None:
        self.data_path = "../data/house_data.csv"
        self.random_seed = 373
        self.test_size = 0.2
        self.n_iter_search = 10
        self.cv_folds = 5