import sys
import os
import argparse

import comet_functions as interactive

descriptions = {
    "oEffect": "The effect of the event on participants besides PersonX might be: ",
    "oReact": "Other participants may react to the event in this way: ",
    "oWant": "After the event, other participants may want: ",
    "xAttr": "Because of the event, we can say that PersonX is: ",
    "xEffect": "The effect of the event on PersonX might be: ",
    "xIntent": "The intent of PersonX in participating in this event is: ",
    "xNeed": "To participate in the event, PersonX needs: ",
    "xReact": "PersonX may react to the event in this way: ",
    "xWant": "After the event, PersonX may want: ",
}

sampling_mapping = {
    "b10": "beam-10",
    "b5": "beam-5",
    "g": "greedy"
}


def parse_input_string(string):
    objects = string.split("|")
    categories = objects[1]

    if not categories or categories == "all":
        final_categories = list(descriptions.keys())
    else:
        final_categories = categories.split(",")

    sampling = sampling_mapping[objects[2]]
    sequence = objects[0]

    return sequence, final_categories, sampling


def format_output_string(text_sequence, sequences):
        print_string = []

        print_string.append("<h3>{}</h3>".format(text_sequence))

        for category, stuff in sequences.items():
            print_string.append("<b>{}</b>".format(descriptions[category]))
            for i, sequence in enumerate(stuff["beams"]):
                print_string.append("({}) {}".format(i + 1, sequence))
            print_string.append("")
        print_string.append("")
        return "<br>".join(print_string)


class DemoModel(object):
    def __init__(self, model_file, vocabulary_path="model/"):
        opt, state_dict, vocab = interactive.load_model_file(model_file)
        # print(opt)
        data_loader, text_encoder = interactive.load_data(
            "atomic", opt, vocab, vocabulary_path)

        self.opt = opt
        self.data_loader = data_loader
        self.text_encoder = text_encoder

        n_ctx = data_loader.max_event + data_loader.max_effect
        n_vocab = len(text_encoder.encoder) + n_ctx

        model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

        self.model = model

    def predict(self, text_sequence, categories, sampling_algorithm, verbose=True):
        sampler = interactive.set_sampler(
            self.opt, sampling_algorithm, self.data_loader)

        sequences = interactive.get_atomic_sequence(
            text_sequence, self.model, sampler, self.data_loader,
            self.text_encoder, categories)

        return sequences

    def getOutput(self, text_string):
        print(text_string)
        text_sequence, categories, sampling_algorithm = parse_input_string(text_string)

        model_output_sequences = self.predict(
            text_sequence, categories, sampling_algorithm, verbose=True)

        return format_output_string(text_sequence, model_output_sequences)


if __name__ == "__main__":
    # from server import run
    # sys.path.append("ATOMIC_NLGWebsite")
    # sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")

    args = parser.parse_args()

    interactive.set_compute_mode(args.device)

    myNLGmodel = DemoModel(args.model_file)

    run(nlg=myNLGmodel)
