import torch
import sys
import os
import time
import comet_config as cfg
import comet_data as data
import comet_models as models
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

import comet_data as data
import comet_config as cfg

##############################################################################
#                                       BATCH
##############################################################################

def prepare_position_embeddings(opt, encoder_vocab, sequences):
    vocab_size = len(encoder_vocab)
    num_positions = sequences.size(-2)
    position_embeddings = torch.LongTensor(
        range(vocab_size, vocab_size + num_positions)).to(sequences.device)
    sequences = sequences.repeat(1, 1, 2)
    sequences[:, :, 1] = position_embeddings
    return sequences

def batch_atomic_generate(opt, nums, losses, batch_variables, eval_mode=False):
    data_loader = batch_variables["data"]
    model = batch_variables["model"]
    split = batch_variables["split"]

    batch, reset = data_loader.sample_batch(split, bs=opt.train.dynamic.bs)

    input_ = prepare_position_embeddings(
        opt, data_loader.vocab_encoder, batch["sequences"].unsqueeze(-1))
    attention_mask = batch["attention_mask"]
    loss_mask = batch["loss_mask"]

    targets = input_.squeeze(0)[:, 1:, 0].contiguous().view(-1)

    loss, dist = mle_steps(
        opt.net.model, model, input_[:, :-1, :], targets,
        attention_mask[:, :-1], loss_reduction="none")

    # Set loss name
    micro_name = "total_micro"
    macro_name = "total_macro"

    length = loss_mask.sum(1)
    bs = input_.size(0)

    final_loss = (loss * loss_mask).sum(1)

    update_generation_losses(losses, nums, micro_name, macro_name, bs,
                             length, (loss * loss_mask).sum(1), split)

    final_loss = final_loss / length

    outputs = {"loss": final_loss.sum(), "nums": nums, "reset": reset}

    return outputs


def batch_conceptnet_generate(opt, nums, losses, batch_variables,
                              eval_mode=False, tracking_mode=False):
    data_loader = batch_variables["data"]
    model = batch_variables["model"]
    split = batch_variables["split"]
    category = batch_variables["category"]

    batch, reset = data_loader.sample_batch(
        split, bs=opt.train.dynamic.bs, cat=category)

    input_ = prepare_position_embeddings(
        opt, data_loader.vocab_encoder, batch["sequences"].unsqueeze(-1))
    attention_mask = batch["attention_mask"]
    loss_mask = batch["loss_mask"]

    targets = input_.squeeze(0)[:, 1:, 0].contiguous().view(-1)

    loss, dist = mle_steps(
        opt.net.model, model, input_[:, :-1, :], targets,
        attention_mask[:, :-1], loss_reduction="none")

    # Set loss name
    if not eval_mode or batch_variables["category"] == "positive":
        micro_name = "total_micro"
        macro_name = "total_macro"
    else:
        micro_name = "negative_micro"
        macro_name = "negative_macro"

    length = loss_mask.sum(1)
    bs = input_.size(0)

    final_loss = (loss * loss_mask).sum(1)

    update_generation_losses(losses, nums, micro_name, macro_name, bs,
                             length, (loss * loss_mask).sum(1), split)

    final_loss = final_loss / length

    outputs = {"loss": final_loss.sum(), "nums": nums, "reset": reset}

    if tracking_mode:
        outputs["tracking"] = final_loss.squeeze().tolist()

    return outputs


def mle_steps(key, model, input_, targets, attention_mask,
              loss_reduction="mean", i=None):
    word_acts = decode(model, input_.unsqueeze(1),
                       attention_mask, i)

    word_dist = modify_output_for_loss_fn(
        "nll", word_acts, dim=-1)

    # Compute losses
    loss = F.nll_loss(
        word_dist.view(-1, word_dist.size(-1)),
        targets, reduction=loss_reduction)

    if loss_reduction != "mean":
        return loss.view(word_dist.size(0), -1), word_dist
    else:
        return loss, word_dist


def decode(model, input_, attention_mask, i=None):
    return model(input_, sequence_mask=attention_mask)


def update_generation_losses(losses, nums, micro, macro, bs,
                             length, loss, split):
    if split == "train":
        update_generation_losses(
            losses, nums, micro, macro, bs, length, loss)
    else:
        update_generation_losses(
            losses, nums, micro, macro, bs, length, loss)

def make_sampler(sampler_type, opt, *args, **kwargs):
    print("Initializing Greedy Sampler")
    return GreedySampler(opt, *args, **kwargs)

class Sampler():
    def __init__(self, opt, data_loader, batch_mode=False):
        # Token on which to end sampling
        self.end_token = data_loader.vocab_encoder[data.end_token]

        self.opt = opt

    def generate_sequence(self, batch, model):
        raise


class GreedySampler(Sampler):
    def __init__(self, opt, data_loader, batch_mode=True):
        super(GreedySampler, self).__init__(opt, data_loader)

    def append_batch(self, X, next_idx, mask):
        next_pos = X[:, -1:, 1] + 1
        next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
        next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
        return torch.cat((X, next_x), 1), next_mask

    def generate_sequence(self, batch, model, data_loader, start_idx, end_len):
        XMB = batch["sequences"][:, :start_idx]
        MMB = batch["attention_mask"][:, :start_idx]

        XMB = prepare_position_embeddings(
            self.opt, data_loader.vocab_encoder, XMB.unsqueeze(-1))

        lm_probs = F.log_softmax(model(
            XMB.unsqueeze(1), sequence_mask=MMB), dim=-1)

        values, indices = lm_probs[:, -1, :].max(dim=-1)
        seqs = indices.clone().unsqueeze(1)

        loss = values
        counts = 1
        next_pos = XMB[:, -1:, 1] + 1
        next_x = torch.cat((indices.view(-1, 1), next_pos), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)

        # Sample from top k

        for _ in range(self.opt.eval.smax):
            lm_probs = F.log_softmax(model(
                XMB.unsqueeze(1), sequence_mask=MMB), dim=-1)

            # Sample from top k
            values, next_idx = lm_probs[:, -1, :].max(dim=-1)

            loss += values
            counts += 1

            next_idx = next_idx.unsqueeze(1)

            seqs = torch.cat([seqs, next_idx], 1)

            if (next_idx.item() == self.end_token) or (_ == end_len - 1):
                break

            XMB, MMB = self.append_batch(XMB, next_idx, MMB)

        beams = []

        for beam in seqs:
            beams.append(" ".join("".join(
                [data_loader.vocab_decoder[tok.item()].replace(
                    '</w>', ' ').replace('\n', '')
                 for tok in beam if tok != self.end_token]).split()))

        sampling_result = {
            "sequence": beams[0],
            "beams": beams,
            "beam_losses": [loss.item()],
            "loss": loss.item(),
            "beam_lengths": [counts],
            "length": counts
        }

        return sampling_result


class TopKSampler(Sampler):
    def __init__(self, opt, data_loader, batch_mode=True):
        super(TopKSampler, self).__init__(opt, data_loader)

    def append_batch(self, X, next_idx, mask):
        next_pos = X[:, -1:, 1] + 1
        next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
        next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
        return torch.cat((X, next_x), 1), next_mask

    def generate_sequence(self, batch, model, data_loader, start_idx, end_len):
        # start_idx = context_size_event + 1
        # start_idx = max_e1 + max_r
        # end_idx = context_size_effect - 1
        # end_idx = max_e2
        XMB = batch["sequences"][:, :start_idx]
        MMB = batch["attention_mask"][:, :start_idx]

        XMB = prepare_position_embeddings(
            self.opt, data_loader.vocab_encoder, XMB.unsqueeze(-1))

        lm_probs = F.log_softmax(model(
            XMB.unsqueeze(1), sequence_mask=MMB), dim=-1)

        values, indices = lm_probs[:, -1, :].topk(self.opt.eval.k)
        seqs = indices.t().clone()

        losses = - values.view(-1, 1)

        ended = (seqs == self.end_token).float()
        counts = (1 - ended)
        XMB = XMB.repeat(self.opt.eval.k, 1, 1)
        MMB = MMB.repeat(self.opt.eval.k, 1)
        next_pos = XMB[:, -1:, 1] + 1
        next_x = torch.cat((indices.view(self.opt.eval.k, -1), next_pos), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)

        # Sample from top k

        for _ in range(end_len):
            lm_probs = F.log_softmax(model(
                XMB.unsqueeze(1), sequence_mask=MMB), dim=-1)

            # Sample from top k
            values, indices = lm_probs[:, -1, :].topk(self.opt.eval.k)
            choice = torch.multinomial(values.exp(), 1)
            next_idx = indices.gather(-1, choice)

            ended = ended + (next_idx == self.end_token).float() * (1 - ended)

            next_idx = next_idx * (1 - ended).long() + ended.long() * self.end_token

            counts += (1 - ended)

            seqs = torch.cat([seqs, next_idx], 1)

            if ended.sum().item() == self.opt.eval.k:
                break

            losses -= values.gather(-1, choice) * (1 - ended)

            XMB, MMB = self.append_batch(XMB, next_idx, MMB)

        beams = []

        for beam in seqs:
            beams.append(" ".join("".join(
                [data_loader.vocab_decoder[tok.item()].replace(
                    '</w>', ' ').replace('\n', '')
                 for tok in beam if tok != self.end_token]).split()))

        sampling_result = {
            "sequence": beams[0],
            "beams": beams,
            "beam_losses": losses.squeeze().tolist(),
            "loss": losses[0].item(),
            "beam_lengths": counts.long().squeeze().tolist(),
            "length": counts[0].long().item()
        }

        return sampling_result


class BeamSampler(TopKSampler):
    def __init__(self, opt, data_loader, batch_mode=True, scorer=None):
        super(BeamSampler, self).__init__(opt, data_loader, batch_mode)

        self.kill_mask = torch.ones(opt.eval.bs, opt.eval.bs).to(cfg.device) * 9000
        self.kill_mask[:, 0] = 0

    def make_batch(self, X):
        X = np.array(X)
        assert X.ndim in [1, 2]
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        pos_enc = np.arange(n_vocab + n_special, n_vocab + n_special + X.shape[-1])
        pos_enc = np.expand_dims(pos_enc, axis=0)
        batch = np.stack([X, pos_enc], axis=-1)
        batch = torch.tensor(batch, dtype=torch.long).to(device)
        return batch

    def append_batch(self, X, beam_toks, mask):
        next_pos = X[:, -1:, 1] + 1
        next_x = torch.cat((beam_toks.unsqueeze(1), next_pos), -1).unsqueeze(1)
        next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
        return torch.cat((X, next_x), 1), next_mask

    def generate_sequence(self, batch, model, data_loader, start_idx, end_len):
        # start_idx = context_size_event + 1
        # start_idx = max_e1 + max_r
        # end_idx = context_size_effect - 1
        # end_idx = max_e2
        XMB = batch["sequences"][:, :start_idx]
        MMB = batch["attention_mask"][:, :start_idx]

        XMB = prepare_position_embeddings(
            self.opt, data_loader.vocab_encoder, XMB.unsqueeze(-1))

        tokens = []
        beam_losses = []
        # Beam Search
        beam_lls, beam_toks, beam_seqs = None, None, None
        lm_probs = F.log_softmax(model(
            XMB.unsqueeze(1), sequence_mask=MMB), dim=-1)
        dist = lm_probs[:, -1, :].squeeze()
        beam_lls, beam_toks = dist.topk(self.opt.eval.bs)
        beam_losses.append(beam_lls)

        ended = (beam_toks == self.end_token).float()
        counts = (2 - ended)
        beam_toks = beam_toks.unsqueeze(1)
        beam_seqs = beam_toks.clone()
        XMB = XMB.repeat(self.opt.eval.bs, 1, 1)
        MMB = MMB.repeat(self.opt.eval.bs, 1)
        next_pos = XMB[:, -1:, 1] + 1
        next_x = torch.cat((beam_toks, next_pos), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)

        for _ in range(end_len):

            # Compute distribution for current beam
            lm_probs = F.log_softmax(model(
                XMB.unsqueeze(1), sequence_mask=MMB), dim=-1)
            dist = lm_probs[:, -1, :].squeeze()

            # get hypothesis tokens for distribution
            hyp_beam_lls, hyp_beam_toks = dist.topk(self.opt.eval.bs)

            # Compute masks and expand beam
            expanded_ended = ended.unsqueeze(1).repeat(1, self.opt.eval.bs)
            hypothesis_mask = expanded_ended * self.kill_mask + (1 - expanded_ended)
            current_beam_lls = beam_lls.unsqueeze(1).repeat(
                1, self.opt.eval.bs).view(self.opt.eval.bs**2)

            # Compute losses of hypotheses, masking those that have ended
            hyp_beam_lls = (hyp_beam_lls.view(self.opt.eval.bs**2) *
                            hypothesis_mask.view(-1)) + current_beam_lls

            # Get normalizer for sequences
            temp_counts = counts.unsqueeze(1).repeat(1, self.opt.eval.bs).view(
                self.opt.eval.bs ** 2)

            # Select best beams with lowest aggregate loss
            beam_lls, top_beam_idxs = (hyp_beam_lls / temp_counts).topk(self.opt.eval.bs)

            # Update placements in beam based on selecetion
            beam_losses = [i.index_select(0, top_beam_idxs // self.opt.eval.bs)
                           for i in beam_losses]
            ended = ended.index_select(0, top_beam_idxs // self.opt.eval.bs)
            counts = temp_counts.index_select(0, top_beam_idxs)

            # Save beam losses
            beam_losses.append(beam_lls * counts)

            # Update beam tokens
            ended_mask = (1 - ended).long()
            end_replacement = (self.end_token * ended).long()
            next_toks = hyp_beam_toks.view(-1)[top_beam_idxs]
            beam_toks = next_toks * ended_mask + end_replacement

            # Update ended and counts
            ended = ended + (beam_toks == self.end_token).float() * (1 - ended)
            counts = counts + (1 - ended)

            # Update beam sequences
            beam_seqs = beam_seqs.t().repeat(self.opt.eval.bs, 1).t().contiguous().view(
                self.opt.eval.bs**2, -1)[top_beam_idxs]
            beam_seqs = torch.cat((beam_seqs, beam_toks.unsqueeze(1)), dim=1)

            XMB = XMB.transpose(0, 1).transpose(1, 2).repeat(
                self.opt.eval.bs, 1, 1).transpose(2, 1).transpose(
                1, 0).contiguous().view(
                self.opt.eval.bs**2, XMB.size(1), XMB.size(2))[top_beam_idxs]

            XMB, MMB = self.append_batch(XMB, beam_toks, MMB)

            if (beam_toks == self.end_token).sum().item() == self.opt.eval.bs:
                break

        beams = []

        for beam in beam_seqs:
            beams.append(" ".join("".join(
                [data_loader.vocab_decoder[tok.item()].replace(
                    '</w>', ' ').replace('\n', '')
                 for tok in beam if tok != self.end_token]).split()))

        sampling_result = {
            "sequence": beams[0],
            "beams": beams,
            "beam_losses": beam_lls.tolist(),
            "loss": beam_lls[0].item(),
            "beam_lengths": counts.tolist(),
            "length": counts[0].item()
        }

        return sampling_result

def set_compute_mode(mode):
    if mode == "auto" and torch.cuda.is_available():
        cfg.device = "cuda"
    elif mode.isdigit():
        cfg.device = int(mode)
    elif mode in ["cpu", "cuda"]:
        cfg.device = mode
    else:
        raise
    print("Pushing units to: {}".format(cfg.device))


class BaseDataLoader(object):
    def __init__(self, opt, vocab):
        self.vocab_encoder = vocab
        self.vocab_decoder = {j: i for i, j in self.vocab_encoder.items()}


class ConceptNetBaseDataLoader(BaseDataLoader):
    def __init__(self, opt, vocab):
        super(ConceptNetBaseDataLoader, self).__init__(opt, vocab)
        self.max_e1 = opt.data.get("maxe1", 10)
        self.max_e2 = opt.data.get("maxe2", 15) + 1
        self.max_r = opt.data.get("maxr", 5)


class AtomicBaseDataLoader(BaseDataLoader):
    def __init__(self, opt, vocab):
        super(AtomicBaseDataLoader, self).__init__(opt, vocab)
        self.max_event = opt.data.get("maxe1", 17)
        self.max_effect = opt.data.get("maxe2", 34) + 1


def load_model_file(model_file):
    model_stuff = data.load_checkpoint(model_file)
    opt = utils.convert_nested_dict_to_DD(model_stuff["opt"])

    state_dict = model_stuff["state_dict"]
    vocab = model_stuff['vocab']
    return opt, state_dict, vocab


def load_data(dataset, opt, vocab, vocabulary_path):
    if dataset == "atomic":
        data_loader = load_atomic_data(opt, vocab)
    elif dataset == "conceptnet":
        data_loader = load_conceptnet_data(opt, vocab)

    # Initialize TextEncoder
    encoder_path = vocabulary_path + "encoder_bpe_40000.json"
    bpe_path = vocabulary_path + "vocab_40000.bpe"
    text_encoder = utils.TextEncoder(encoder_path, bpe_path)
    text_encoder.encoder = data_loader.vocab_encoder
    text_encoder.decoder = data_loader.vocab_decoder

    return data_loader, text_encoder


def load_atomic_data(opt, vocab):
    # path = "data/atomic/processed/generation/{}.pickle".format(
    #     utils.make_name_string(opt.data))
    # data_loader = data.make_data_loader(opt, opt.data.categories)
    # loaded = data_loader.load_data(path)

    data_loader = AtomicBaseDataLoader(opt, vocab)

    return data_loader


def load_conceptnet_data(opt, vocab):
    # path = "data/conceptnet/processed/generation/{}.pickle".format(
    # utils.make_name_string(opt.data))
    # data_loader = data.make_data_loader(opt)
    # loaded = data_loader.load_data(path)
    data_loader = ConceptNetBaseDataLoader(opt, vocab)
    return data_loader


def make_model(opt, n_vocab, n_ctx, state_dict):
    model = models.make_model(
        opt, n_vocab, n_ctx, None, load=False,
        return_acts=True, return_probs=False)
    model.load_state_dict(state_dict)
    model.to(cfg.device)
    model.eval()

    return model


def set_sampler(opt, sampling_algorithm, data_loader):
    if "beam" in sampling_algorithm:
        opt.eval.bs = int(sampling_algorithm.split("-")[1])
        sampler = BeamSampler(opt, data_loader)
    elif "topk" in sampling_algorithm:
        # print("Still bugs in the topk sampler. Use beam or greedy instead")
        # raise NotImplementedError
        opt.eval.k = int(sampling_algorithm.split("-")[1])
        sampler = TopKSampler(opt, data_loader)
    else:
        sampler = GreedySampler(opt, data_loader)

    return sampler


def get_atomic_sequence(input_event, model, sampler, data_loader, text_encoder, category):
    if isinstance(category, list):
        outputs = {}
        for cat in category:
            new_outputs = get_atomic_sequence(
                input_event, model, sampler, data_loader, text_encoder, cat)
            outputs.update(new_outputs)
        return outputs
    elif category == "all":
        outputs = {}

        # start = time.time()

        for category in data.atomic_data.all_categories:
            new_outputs = get_atomic_sequence(
                input_event, model, sampler, data_loader, text_encoder, category)
            outputs.update(new_outputs)

        # end = time.time()
        # print("total time for all categories: {} s".format(end - start))
        return outputs
    else:

        sequence_all = {}

        sequence_all["event"] = input_event
        sequence_all["effect_type"] = category

        with torch.no_grad():

            # start = time.time()

            batch = set_atomic_inputs(
                input_event, category, data_loader, text_encoder)

            # end_set = time.time()

            sampling_result = sampler.generate_sequence(
                batch, model, data_loader, data_loader.max_event +
                data.atomic_data.num_delimiter_tokens["category"],
                data_loader.max_effect -
                data.atomic_data.num_delimiter_tokens["category"])

            # end_sample = time.time()
            # print(category)
            # print("Set inputs: {} s".format(end_set - start))
            # print("Sample: {} s".format(end_sample - end_set))

        sequence_all['beams'] = sampling_result["beams"]

        # print_atomic_sequence(sequence_all)

        return {category: sequence_all}


def print_atomic_sequence(sequence_object):
    input_event = sequence_object["event"]
    category = sequence_object["effect_type"]

    print("Input Event:   {}".format(input_event))
    print("Target Effect: {}".format(category))
    print("")
    print("Candidate Sequences:")
    for beam in sequence_object["beams"]:
        print(beam)
    print("")
    print("====================================================")
    print("")


def set_atomic_inputs(input_event, category, data_loader, text_encoder):
    XMB = torch.zeros(1, data_loader.max_event + 1).long().to(cfg.device)
    prefix, suffix = data.atomic_data.do_example(text_encoder, input_event, None, True, None)

    XMB[:, :len(prefix)] = torch.LongTensor(prefix)
    XMB[:, -1] = torch.LongTensor([text_encoder.encoder["<{}>".format(category)]])

    batch = {}
    batch["sequences"] = XMB
    batch["attention_mask"] = data.atomic_data.make_attention_mask(XMB)

    return batch


def get_conceptnet_sequence(e1, model, sampler, data_loader, text_encoder, relation, force=False):
    if isinstance(relation, list):
        outputs = {}

        for rel in relation:
            new_outputs = get_conceptnet_sequence(
                e1, model, sampler, data_loader, text_encoder, rel)
            outputs.update(new_outputs)
        return outputs
    elif relation == "all":
        outputs = {}

        for relation in data.conceptnet_data.conceptnet_relations:
            new_outputs = get_conceptnet_sequence(
                e1, model, sampler, data_loader, text_encoder, relation)
            outputs.update(new_outputs)
        return outputs
    else:

        sequence_all = {}

        sequence_all["e1"] = e1
        sequence_all["relation"] = relation

        with torch.no_grad():
            if data_loader.max_r != 1:
                relation_sequence = data.conceptnet_data.split_into_words[relation]
            else:
                relation_sequence = "<{}>".format(relation)

            batch, abort = set_conceptnet_inputs(
                e1, relation_sequence, text_encoder,
                data_loader.max_e1, data_loader.max_r, force)

            if abort:
                return {relation: sequence_all}

            sampling_result = sampler.generate_sequence(
                batch, model, data_loader,
                data_loader.max_e1 + data_loader.max_r,
                data_loader.max_e2)

        sequence_all['beams'] = sampling_result["beams"]

        print_conceptnet_sequence(sequence_all)

        return {relation: sequence_all}


def set_conceptnet_inputs(input_event, relation, text_encoder, max_e1, max_r, force):
    abort = False

    e1_tokens, rel_tokens, _ = data.conceptnet_data.do_example(text_encoder, input_event, relation, None)

    if len(e1_tokens) >  max_e1:
        if force:
            XMB = torch.zeros(1, len(e1_tokens) + max_r).long().to(cfg.device)
        else:
            XMB = torch.zeros(1, max_e1 + max_r).long().to(cfg.device)
            return {}, True
    else:
        XMB = torch.zeros(1, max_e1 + max_r).long().to(cfg.device)

    XMB[:, :len(e1_tokens)] = torch.LongTensor(e1_tokens)
    XMB[:, max_e1:max_e1 + len(rel_tokens)] = torch.LongTensor(rel_tokens)

    batch = {}
    batch["sequences"] = XMB
    batch["attention_mask"] = data.conceptnet_data.make_attention_mask(XMB)

    return batch, abort


def print_conceptnet_sequence(sequence_object):
    e1 = sequence_object["e1"]
    relation = sequence_object["relation"]

    print("Input Entity:    {}".format(e1))
    print("Target Relation: {}".format(relation))
    print("")
    print("Candidate Sequences:")
    for beam in sequence_object["beams"]:
        print(beam)
    print("")
    print("====================================================")
    print("")


def print_help(data):
    print("")
    if data == "atomic":
        print("Provide a seed event such as \"PersonX goes to the mall\"")
        print("Don't include names, instead replacing them with PersonX, PersonY, etc.")
        print("The event should always have PersonX included")
    if data == "conceptnet":
        print("Provide a seed entity such as \"go to the mall\"")
        print("Because the model was trained on lemmatized entities,")
        print("it works best if the input entities are also lemmatized")
    print("")


def print_relation_help(data):
    print_category_help(data)


def print_category_help(data):
    print("")
    if data == "atomic":
        print("Enter a possible effect type from the following effect types:")
        print("all - compute the output for all effect types {{oEffect, oReact, oWant, xAttr, xEffect, xIntent, xNeed, xReact, xWant}}")
        print("oEffect - generate the effect of the event on participants other than PersonX")
        print("oReact - generate the reactions of participants other than PersonX to the event")
        print("oEffect - generate what participants other than PersonX may want after the event")
    elif data == "conceptnet":
        print("Enter a possible relation from the following list:")
        print("")
        print('AtLocation')
        print('CapableOf')
        print('Causes')
        print('CausesDesire')
        print('CreatedBy')
        print('DefinedAs')
        print('DesireOf')
        print('Desires')
        print('HasA')
        print('HasFirstSubevent')
        print('HasLastSubevent')
        print('HasPainCharacter')
        print('HasPainIntensity')
        print('HasPrerequisite')
        print('HasProperty')
        print('HasSubevent')
        print('InheritsFrom')
        print('InstanceOf')
        print('IsA')
        print('LocatedNear')
        print('LocationOfAction')
        print('MadeOf')
        print('MotivatedByGoal')
        print('NotCapableOf')
        print('NotDesires')
        print('NotHasA')
        print('NotHasProperty')
        print('NotIsA')
        print('NotMadeOf')
        print('PartOf')
        print('ReceivesAction')
        print('RelatedTo')
        print('SymbolOf')
        print('UsedFor')
        print("")
        print("NOTE: Capitalization is important")
    else:
        raise
    print("")

def print_sampling_help():
    print("")
    print("Provide a sampling algorithm to produce the sequence with from the following:")
    print("")
    print("greedy")
    print("beam-# where # is the beam size")
    print("topk-# where # is k")
    print("")
