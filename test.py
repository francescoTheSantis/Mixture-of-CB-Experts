from torch_concepts.data import ToyDataset

class loader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def load_data(self):
        # Load the data
        if self.cfg.dataset.metadata.name in ['xor', 'trigonometry', 'dot', 'checkmark']:
            data = ToyDataset('xor', size=1000, random_state=42)
        print(data.data.shape, data.concept_labels.shape, data.target_labels.shape,
          data.concept_attr_names, data.task_attr_names)
        #return loaded_train, loaded_val, loaded_test

def main():
    from omegaconf import DictConfig
    from omegaconf import OmegaConf
    from src.loaders.dataloader import loader

    cfg = OmegaConf.create({
        'dataset': {
            'metadata': {
                'name': 'xor'
            }
        }
    })

    data_loader = loader(cfg)
    data_loader.load_data()

if __name__ == "__main__":