import os
import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color
from Trainer.single_trainer import SingleTrainer
import torch

from model.FFMSR_fed import MLTRec

from data.dataset import FederatedDataset


# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def finetune(dataset, pretrained_file, plm, fix_enc=True, **kwargs):
    # configurations initialization
    props = ['props/FFMSR.yaml', 'props/finetune.yaml']
    print(props)

    # configurations initialization
    config = Config(model=MLTRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = FederatedDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    if plm == 'DistillBERT':
        config['text_model'] = './distilbert-base-uncased'
    # model loading and initialization
    model = MLTRec(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(model)

    # fix item embedding
    # model.item_embedding.weight.requires_grad = False
    # model training
    # model.global_prompt.requires_grad = True
    trainer = SingleTrainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, test_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Pantry', help='dataset name')
    parser.add_argument('-p', type=str, default='',
                        help='pre-trained model path')
    parser.add_argument('--plm', type=str, default='DistillBERT')
    parser.add_argument('-f', type=bool, default=False)
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(args.d, pretrained_file=args.p, plm=args.plm, fix_enc=args.f)