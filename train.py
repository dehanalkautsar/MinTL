import os, random, argparse, time, logging, json, tqdm
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch

from utils import Vocab, MultiWozReader, CamRestReader, SMDReader
# from damd_multiwoz.eval import MultiWozEvaluator
from evaluator import CamRestEvaluator, SMDEvaluator
from transformers import (AdamW, T5Tokenizer, BartTokenizer, WEIGHTS_NAME,CONFIG_NAME, get_linear_schedule_with_warmup)
from T5 import MiniT5
from MT5 import MiniMT5
from BART import MiniBART

class BartTokenizer(BartTokenizer):
    def encode(self,text,add_special_tokens=False):
        encoded_inputs = self.encode_plus(text,add_special_tokens=False)
        return encoded_inputs["input_ids"]



class Model(object):
    def __init__(self, args, test=False):
        if args.back_bone=="t5":  
            if args.exp_setting=='en':
                self.tokenizer = T5Tokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
                self.model = MiniT5.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            elif args.exp_setting=='bi' or args.exp_setting=='bi-en' or args.exp_setting=='bi-id' or args.exp_setting=='id' or args.exp_setting=='cross':
                self.tokenizer = T5Tokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
                self.model = MiniMT5.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        elif args.back_bone=="bart": #not implemented
            self.tokenizer = BartTokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            self.model = MiniBART.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        vocab = Vocab(self.model, self.tokenizer)
        
        # check what dataset should be used
        if args.dataset == 'multiwoz':
            self.reader = MultiWozReader(vocab,args)
            self.evaluator = MultiWozEvaluator(self.reader) # evaluator class
        elif args.dataset == 'camrest':
            self.reader = CamRestReader(vocab,args)
            self.evaluator = CamRestEvaluator(self.reader)
        elif args.dataset == 'smd':
            self.reader = SMDReader(vocab,args)
            self.evaluator = SMDEvaluator(self.reader)

        self.optim = AdamW(self.model.parameters(), lr=args.lr)
        self.args = args
        self.model.to(args.device)
        

    def load_model(self):
        # model_state_dict = torch.load(checkpoint)
        # start_model.load_state_dict(model_state_dict)
        self.model = type(self.model).from_pretrained(self.args.model_path)
        self.model.to(self.args.device)

    def train(self, cfg, args):
        btm = time.time()
        step = 0
        prev_min_loss = 1000
        print(f"vocab_size:{self.model.config.vocab_size}")
        torch.save(self.args, self.args.model_path + '/model_training_args.bin')
        self.tokenizer.save_pretrained(self.args.model_path)
        self.model.config.to_json_file(os.path.join(self.args.model_path, CONFIG_NAME))
        self.model.train()
        # lr scheduler
        lr_lambda = lambda epoch: self.args.lr_decay ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda)

        for epoch in range(cfg.epoch_num):
            log_loss = 0
            log_dst = 0
            log_resp = 0
            log_cnt = 0
            sw = time.time()
            data_iterator = self.reader.get_batches('train', cfg)
            for iter_num, dial_batch in enumerate(data_iterator):
                py_prev = {'pv_bspn': None}
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num==0)
                    inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.config.decoder_start_token_id)
                    # # #
                    # copy_input = deepcopy(inputs)
                    # for i in range(5):
                    #     for el in copy_input:
                    #         for j in range(len(copy_input[el][i])):
                    #             copy_input[el][i][j] = copy_input[el][i][j] if copy_input[el][i][j] >=0 else 0
                    #     print(f"INPUT_IDS     : {self.tokenizer.decode(copy_input['input_ids'][i])}")
                    #     print(f"RESPONSE      : {self.tokenizer.decode(copy_input['response'][i])}")
                    #     print(f"RESPONSE_INPUT: {self.tokenizer.decode(copy_input['response_input'][i])}")
                    #     print(f"INPUT_POINTER : {self.tokenizer.decode(copy_input['input_pointer'][i])}")
                    #     print(f"MASKS : {copy_input['masks'][i]}")
                    #     print(f"DECODED MASKS : {self.tokenizer.decode(copy_input['masks'][i])}")
                    #     print(f"STATE_INPUT : {self.tokenizer.decode(copy_input['state_input'][i])}")
                    #     print(f"STATE_UPDATE : {self.tokenizer.decode(copy_input['state_update'][i])}")
                    #     print("------"*10)
                    # # #
                    for k in inputs:
                        if k!="turn_domain":
                            inputs[k] = inputs[k].to(self.args.device)

                    if args.exp_setting == 'en':
                        outputs = self.model(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["masks"],
                                        decoder_input_ids=inputs["state_input"],
                                        lm_labels=inputs["state_update"],
                                        # return_dict=False # not using this because not using MT5Mini
                                        )
                    elif args.exp_setting=='bi' or args.exp_setting=='bi-en' or args.exp_setting=='bi-id' or args.exp_setting=='id' or args.exp_setting=='cross':
                        outputs = self.model(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["masks"],
                                            decoder_input_ids=inputs["state_input"],
                                            lm_labels=inputs["state_update"],
                                            return_dict=False
                                            )
                    dst_loss = outputs[0]

                    if args.exp_setting == 'en':
                        outputs = self.model(encoder_outputs=outputs[-1:], #skip loss and logits
                                                attention_mask=inputs["masks"],
                                                decoder_input_ids=inputs["response_input"],
                                                lm_labels=inputs["response"],
                                                # return_dict=False # not using this because not using MT5Mini
                                                )
                    elif args.exp_setting=='bi' or args.exp_setting=='bi-en' or args.exp_setting=='bi-id' or args.exp_setting=='id' or args.exp_setting=='cross':
                        outputs = self.model(encoder_outputs=outputs[-1:], #skip loss and logits
                                                attention_mask=inputs["masks"],
                                                decoder_input_ids=inputs["response_input"],
                                                lm_labels=inputs["response"],
                                                return_dict=False
                                                )
                    resp_loss = outputs[0]
                    # if args.exp_setting=='en':
                    #     outputs = self.model(encoder_outputs=outputs[-1:], #skip loss and logits
                    #                         attention_mask=inputs["masks"],
                    #                         decoder_input_ids=inputs["response_input"],
                    #                         lm_labels=inputs["response"]
                    #                         )
                    #     resp_loss = outputs[0]
                    # elif args.exp_setting=='bi' or args.exp_setting=='bi-en' or args.exp_setting=='bi-id':
                    #     print(outputs.encoder_last_hidden_state.shape)
                    #     # print(outputs.encoder_last_hidden_state[0][0])
                    #     outputs = self.model(encoder_outputs=outputs.encoder_last_hidden_state,
                    #                         attention_mask=inputs["masks"],
                    #                         decoder_input_ids=inputs["response_input"],
                    #                         lm_labels=inputs["response"]
                    #                         )
                    #     resp_loss = outputs.loss

                    py_prev['bspn'] = turn_batch['bspn']

                    total_loss = (dst_loss + resp_loss) / self.args.gradient_accumulation_steps

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                    step+=1
                    log_loss += float(total_loss.item())
                    log_dst +=float(dst_loss.item())
                    log_resp +=float(resp_loss.item())
                    log_cnt += 1

                if (iter_num+1)%cfg.report_interval==0:
                    logging.info(
                            'iter:{} [total|bspn|resp] loss: {:.2f} {:.2f} {:.2f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                           log_loss/(log_cnt+ 1e-8),
                                                                           log_dst/(log_cnt+ 1e-8),log_resp/(log_cnt+ 1e-8),
                                                                           time.time()-btm,
                                                                           turn_num+1))
            epoch_sup_loss = log_loss/(log_cnt+ 1e-8)
            do_test = False
            valid_loss = self.validate(cfg=cfg, do_test=do_test)
            logging.info('epoch: %d, train loss: %.3f, valid loss: %.3f, total time: %.1fmin' % (epoch+1, epoch_sup_loss,
                    valid_loss, (time.time()-sw)/60))

            if valid_loss < prev_min_loss:
                early_stop_count = cfg.early_stop_count
                weight_decay_count = cfg.weight_decay_count
                prev_min_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, WEIGHTS_NAME))
                logging.info('Model saved')
                #self.save_model(epoch)
            else:
                early_stop_count -= 1
                weight_decay_count -= 1
                scheduler.step()
                logging.info('epoch: %d early stop countdown %d' % (epoch+1, early_stop_count))


                if not early_stop_count:
                    self.load_model()
                    print('result preview...')
                    file_handler = logging.FileHandler(os.path.join(self.args.model_path, 'eval_log%s.json'%cfg.seed))
                    logging.getLogger('').addHandler(file_handler)
                    logging.info(str(cfg))
                    self.eval(cfg)
                    return
                # if not weight_decay_count:
                #     self.optim = AdamW(model.parameters(), lr=args.lr)
                #     lr *= cfg.lr_decay
                #     self.optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),
                #                   weight_decay=5e-5)
                #     weight_decay_count = cfg.weight_decay_count
                #     logging.info('learning rate decay, learning rate: %f' % (lr))

        self.load_model()
        print('result preview...')
        file_handler = logging.FileHandler(os.path.join(self.args.model_path, 'eval_log%s.json'%cfg.seed))
        logging.getLogger('').addHandler(file_handler)
        logging.info(str(cfg))
        self.eval(cfg)


    def validate(self, cfg, data='dev', do_test=False):
        self.model.eval()
        valid_loss, count = 0, 0
        data_iterator = self.reader.get_batches(data, cfg)
        result_collection = {}
        for batch_num, dial_batch in enumerate(data_iterator):
            py_prev = {'bspn': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.config.decoder_start_token_id)
                for k in inputs:
                    if k!="turn_domain":
                        inputs[k] = inputs[k].to(self.args.device)
                if self.args.noupdate_dst and (self.args.dataset == 'camrest' or self.args.dataset == 'smd'):
                    dst_outputs, resp_outputs = self.model.inference_sequicity(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], db=inputs["input_pointer"], dataset_type=self.args.dataset, dial_id=inputs["dial_id"])
                elif self.args.noupdate_dst:
                    dst_outputs, resp_outputs = self.model.inference_sequicity(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                else:
                    dst_outputs, resp_outputs = self.model.inference(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                turn_batch['resp_gen'] = resp_outputs
                turn_batch['bspn_gen'] = dst_outputs
                py_prev['bspn'] = dst_outputs

            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        results, _ = self.reader.wrap_result(result_collection, cfg=cfg)
        bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu
        valid_loss = 1.3 - score
        logging.info('validation [CTR] match: %2.5f  success: %2.5f  bleu: %2.5f'%(match, success, bleu))
        self.model.train()
        if do_test:
            print('result preview...')
            self.eval(cfg)
        return valid_loss

    def eval(self, cfg, data='test'):
        self.model.eval()
        self.reader.result_file = None
        result_collection = {}
        data_iterator = self.reader.get_batches(data, cfg)
        for batch_num, dial_batch in tqdm.tqdm(enumerate(data_iterator)):
            py_prev = {'bspn': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.config.decoder_start_token_id)
                for k in inputs:
                    if k!="turn_domain":
                        inputs[k] = inputs[k].to(self.args.device)
                if self.args.noupdate_dst and (self.args.dataset == 'camrest' or self.args.dataset == 'smd'):
                    dst_outputs, resp_outputs = self.model.inference_sequicity(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], db=inputs["input_pointer"], dataset_type=self.args.dataset, dial_id=inputs["dial_id"])
                elif self.args.noupdate_dst:
                    dst_outputs, resp_outputs = self.model.inference_sequicity(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                else:
                    dst_outputs, resp_outputs = self.model.inference(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                turn_batch['resp_gen'] = resp_outputs
                turn_batch['bspn_gen'] = dst_outputs
                py_prev['bspn'] = dst_outputs
             
            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        results, field = self.reader.wrap_result(result_collection, cfg=cfg)

        metric_results = self.evaluator.run_metrics(results, eval_act=False)
        metric_field = list(metric_results[0].keys()) 

        if self.args.dataset == 'multiwoz':
            self.reader.save_result('w', results, field)
            req_slots_acc = metric_results[0]['req_slots_acc']
            info_slots_acc = metric_results[0]['info_slots_acc']
            self.reader.save_result('w', metric_results, metric_field,
                                                write_title='EVALUATION RESULTS:')
            self.reader.save_result('a', [info_slots_acc], list(info_slots_acc.keys()),
                                                write_title='INFORM ACCURACY OF EACH SLOTS:')
            self.reader.save_result('a', [req_slots_acc], list(req_slots_acc.keys()),
                                                write_title='REQUEST SUCCESS RESULTS:')
            self.reader.save_result('a', results, field+['wrong_domain', 'wrong_act', 'wrong_inform'],
                                                write_title='DECODED RESULTS:')
            self.reader.save_result_report(metric_results)
        else: # camrest or smd
            self.reader.save_result('w', metric_results, metric_field, cfg=cfg,
                                                write_title='EVALUATION RESULTS:')
            self.reader.save_result('a', results, field, cfg=cfg, write_title='DECODED RESULTS')


        # self.reader.metric_record(metric_results)
        self.model.train()
        return None

    def lexicalize(self, result_path,output_path):
        self.reader.relex(result_path,output_path)

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True


    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = int(sum([np.prod(p.size()) for p in module_parameters]))

        print('total trainable params: %d' % param_cnt)
        return param_cnt


def parse_arg_cfg(args, cfg):
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k=='cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Accumulate gradients on several steps")
    parser.add_argument("--pretrained_checkpoint", type=str, default="t5-small", help="t5-small, t5-base, bart-large")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--context_window", type=int, default=5, help="how many previous turns for model input")
    parser.add_argument("--lr_decay", type=float, default=0.8, help="Learning rate decay")
    parser.add_argument("--noupdate_dst", action='store_true', help="dont use update base DST")
    parser.add_argument("--back_bone", type=str, default="t5", help="choose t5 or bart")
    parser.add_argument("--dataset", type=str, default="multiwoz", help="choose the dataset: multiwoz, camrest, or smd")
    parser.add_argument("--exp_setting", type=str, default="en", help="choose the experiment setting between en,id,cross,bi,bi-en,and bi-id")
    #parser.add_argument("--dst_weight", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()

    if args.dataset == 'multiwoz':
        from damd_multiwoz.config import global_config as cfg
    elif args.dataset == 'camrest':
        if args.exp_setting == 'en':
            from config import camrest_config_en as cfg
        elif args.exp_setting == 'id':
            from config import camrest_config_id as cfg
        elif args.exp_setting == 'cross':
            from config import camrest_config_cross as cfg
        elif args.exp_setting == 'bi':
            from config import camrest_config_bi as cfg
        elif args.exp_setting == 'bi-en':
            from config import camrest_config_bien as cfg
        elif args.exp_setting == 'bi-id':
            from config import camrest_config_biid as cfg
    elif args.dataset == 'smd':
        if args.exp_setting == 'en':
            from config import smd_config_en as cfg
        elif args.exp_setting == 'id':
            from config import smd_config_id as cfg
        elif args.exp_setting == 'cross':
            from config import smd_config_cross as cfg
        elif args.exp_setting == 'bi':
            from config import smd_config_bi as cfg
        elif args.exp_setting == 'bi-en':
            from config import smd_config_bien as cfg
        elif args.exp_setting == 'bi-id':
            from config import smd_config_biid as cfg

    cfg.mode = args.mode
    cfg.exp_setting = args.exp_setting
    if args.mode == 'test' or args.mode == 'relex':
        parse_arg_cfg(args, cfg)
        cfg_load = json.loads(open(os.path.join(args.model_path, 'exp_cfg.json'), 'r').read())
        for k, v in cfg_load.items():
            if k in ['mode', 'cuda', 'cuda_device', 'eval_per_domain', 'use_true_pv_resp',
                        'use_true_prev_bspn','use_true_prev_aspn','use_true_curr_bspn','use_true_curr_aspn',
                        'name_slot_unable', 'book_slot_unable','count_req_dials_only','log_time', 'model_path',
                        'result_path', 'model_parameters', 'multi_gpu', 'use_true_bspn_for_ctr_eval', 'nbest',
                        'limit_bspn_vocab', 'limit_aspn_vocab', 'same_eval_as_cambridge', 'beam_width',
                        'use_true_domain_for_ctr_eval', 'use_true_prev_dspn', 'aspn_decode_mode',
                        'beam_diverse_param', 'same_eval_act_f1_as_hdsa', 'topk_num', 'nucleur_p',
                        'act_selection_scheme', 'beam_penalty_type', 'record_mode']:
                continue
            setattr(cfg, k, v)
            cfg.result_path = os.path.join(args.model_path, 'result.csv')
    else:
        parse_arg_cfg(args, cfg)
        if args.model_path=="":
            args.model_path = 'experiments/{}_sd{}_lr{}_bs{}_sp{}_dc{}_cw{}_model_{}_noupdate{}_{}/'.format('-'.join(cfg.exp_domains), cfg.seed, args.lr, cfg.batch_size,
                                                                                            cfg.early_stop_count, args.lr_decay, args.context_window, args.pretrained_checkpoint, args.noupdate_dst, args.fraction)
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        cfg.result_path = os.path.join(args.model_path, 'result.csv')
        cfg.eval_load_path = args.model_path

    cfg._init_logging_handler(args.mode)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    #cfg.model_parameters = m.count_params()
    logging.info(str(cfg))

    if args.mode == 'train':
        with open(os.path.join(args.model_path, 'exp_cfg.json'), 'w') as f:
            json.dump(cfg.__dict__, f, indent=2)
        m = Model(args)
        m.train(cfg,args)
    elif args.mode == 'test':
        m = Model(args,test=True)
        m.eval(cfg=cfg, data='test')
    elif args.mode == 'relex':
        m = Model(args,test=True)
        output_path = os.path.join(args.model_path, 'generation.csv')
        m.lexicalize(cfg.result_path,output_path)


if __name__ == '__main__':
    main()
