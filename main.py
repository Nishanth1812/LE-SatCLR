def main():
    import argparse
    from src.config import Config
    from src.training import train
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['simclr','probe','finetune','baseline'],default='simclr')
    parser.add_argument('--data-root',default='Dataset/EuroSAT_RGB')
    parser.add_argument('--output-root',default='outputs')
    parser.add_argument('--encoder-checkpoint')
    parser.add_argument('--tracking-uri')
    parser.add_argument('--epochs',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=0)
    parser.add_argument('--label-percent',type=int,choices=[1,10],default=1)
    parser.add_argument('--max-batches',type=int,default=0)
    parser.add_argument('--policy',choices=['standard','satellite'],default='standard')
    parser.add_argument('--ssl-scope',choices=['all','train'],default='all')
    args = vars(parser.parse_args())
    runtime = {key:args.pop(key) for key in ['data_root','output_root','encoder_checkpoint','tracking_uri']}
    from src.config import resolve_batch_size, resolve_epochs
    args['epochs'] = resolve_epochs(args['stage'],args['epochs'])
    args['batch_size'] = resolve_batch_size(args['stage'],args['batch_size'])
    train(Config(**args),**runtime)


if __name__ == "__main__":
    main()
