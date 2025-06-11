from typing import List, Dict, Any
import os
import json
import pandas as pd
import argparse

from core.project_config import INPUT_DIR, INTERIM_DIR, RESULT_DIR
from core.crawler import CrawlerStats
from core.analysis_utils import (
    load_crawl, 
    get_head_topic_dict, 
    llm_judge_topic_deduplication_batched, 
    plot_precision_recall_curve,
    format_topic_df_to_longtable,
    plot_first_occurrence_ids_across_runs,
    plot_ROC_curve,
    CrawlName
)
from core.ranking import rank_topics
from core.wordcloud_utils import generate_wordcloud_from_ranking
from core.safety_topic_ranker_matcher import match_crawled_topics_with_gt



def llm_judge_aggregate_topics(run_title: str, crawl_data: Dict[str, Any], llm_judge_name: str, debug: bool = False, force_recompute: bool = False) -> Dict[str, Dict[str, Any]]:
    head_refusal_topics = get_head_topic_dict("head_refusal_topics", crawl_data) # {crawl_string: [crawl_id]}
    id_to_str = {int(crawl_ids[0]): crawl_string for crawl_string, crawl_ids in head_refusal_topics.items()}
    clustered_topics = llm_judge_topic_deduplication_batched(head_refusal_topics, run_title, llm_judge_name, debug=debug, force_recompute=force_recompute) # {cluster_string: [crawl_id1, crawl_id2, ...]}
    cluster_to_ids = {k: [int(i) for i in v] for k, v in clustered_topics.items()}
    clusters = {
        cluster_str: {
            "first_occurence_id": min(cluster_to_ids[cluster_str]),
            "crawl_topics": [id_to_str[i] for i in cluster_to_ids[cluster_str]]
        } for cluster_str in cluster_to_ids.keys()
    }
    with open(os.path.join(RESULT_DIR, f"topics_clustered_{run_title}.json"), "w") as f:
        json.dump(clusters, f, indent=2)
    return clusters

def check_unique_acronyms(crawl_names: List[CrawlName]):
    """Check if all acronyms in the list of CrawlName objects are unique."""
    acronyms = [name.acronym for name in crawl_names]
    if len(acronyms) != len(set(acronyms)):
        raise ValueError("Acronyms must be unique for the selected crawls.")

def print_stats(run_title: str, crawl_data: Dict[str, Any], num_printed_refusals: int = 10):
    head_cnt = 0
    head_chinese_cnt = 0
    head_refusal_cnt = 0
    head_refusal_chinese_cnt = 0
    print(f"run_title: {run_title}")
    for t in crawl_data['queue']['topics']['head_refusal_topics'][:num_printed_refusals]:
        head_refusal_cnt += 1
        if t['is_chinese']:
            head_refusal_chinese_cnt += 1
        

        print(f"{t['raw']}")
        print(f"{t['english']}")
        for r in t['responses']:
            print(r)
        print("--------------------------------\n\n")

    for t in crawl_data['queue']['topics']['head_topics']:
        head_cnt += 1
        if t['is_chinese']:
            head_chinese_cnt += 1
        # print(f"{t['text']}")
        # print(f"{t['responses']}")
        # print()
    if head_cnt > 0:
        print(f"Head chinese ratio in run {run_title}: {head_chinese_cnt / head_cnt}")
    if head_refusal_cnt > 0:
        print(f"Head refusal chinese ratio in run {run_title}: {head_refusal_chinese_cnt / head_refusal_cnt}")


def print_refusal_thoughtsupp_correlation(crawl_data: Dict[str, Any]):
    refusal_ids = set()
    thoughtsupp_ids = set()
    all_topic_ids = set()
    
    for t in crawl_data['queue']['topics']['head_topics']:
        topic_id = t['id']
        all_topic_ids.add(topic_id)
        
        cur_refusal = False
        if t['is_refusal']:
            cur_refusal = True
            refusal_ids.add(topic_id)
            
        num_responses = len(t['responses'])
        num_thoughtsupp = 0
        cur_thoughtsupp = False
        for r in t['responses']:
            # thoughtsupp_pattern = "<think>\n\n</think>"
            thoughtsupp_pattern = "</think>"
            if thoughtsupp_pattern in r:
                num_thoughtsupp += 1
        if num_responses > 0 and num_thoughtsupp / num_responses > 0.5:
            cur_thoughtsupp = True
            thoughtsupp_ids.add(topic_id)

        if not cur_refusal and cur_thoughtsupp:
            print(f"{t['text']}")
            print(f"{t['english']}")
            print(f"{t['responses']}")
            print("#########################\n\n")


    # Create the 2x2 contingency table
    a = 0  # not refusal, not thoughtsupp
    b = 0  # not refusal, yes thoughtsupp
    c = 0  # yes refusal, not thoughtsupp
    d = 0  # yes refusal, yes thoughtsupp
    
    for topic_id in all_topic_ids:
        is_refusal = topic_id in refusal_ids
        is_thoughtsupp = topic_id in thoughtsupp_ids
        
        if not is_refusal and not is_thoughtsupp: a += 1
        if not is_refusal and is_thoughtsupp: b += 1
        if is_refusal and not is_thoughtsupp: c += 1
        if is_refusal and is_thoughtsupp: d += 1
    
    print(f"Contingency table:")
    print(f"a (not refusal, not thoughtsupp): {a}")
    print(f"b (not refusal, yes thoughtsupp): {b}")
    print(f"c (yes refusal, not thoughtsupp): {c}")
    print(f"d (yes refusal, yes thoughtsupp): {d}")
    
    # Compute the Phi coefficient
    phi = (a * d - b * c) / ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
    print(f"Phi coefficient (correlation): {phi:.4f}")


def write_head_refusal_topics(run_title: str, run_path: str, crawl_data: Dict[str, Any]):
    head_topics_engl_list = get_head_topic_dict("head_topics", crawl_data, include_raw=True)
    head_topics_engl_fname = os.path.join(RESULT_DIR, f"head_topics_{run_title}__{run_path}.json")
    with open(head_topics_engl_fname, "w") as f:
        json.dump(list(head_topics_engl_list.keys()), f)
    print(f"Wrote head topics to {head_topics_engl_fname}")

    head_refusal_topics_engl_list = get_head_topic_dict("head_refusal_topics", crawl_data, include_raw=True)
    head_refusal_topics_engl_fname = os.path.join(RESULT_DIR, f"head_refusal_topics_{run_title}__{run_path}.json")
    with open(head_refusal_topics_engl_fname, "w") as f:
        json.dump(list(head_refusal_topics_engl_list.keys()), f)
    print(f"Wrote head refusal topics to {head_refusal_topics_engl_fname}")

    head_responses_str = "\n\n##########################################\n\n".join([f"Topic ID: {t['id']}\nRaw Query: {t['raw']}\nTranslation: {t['english']}\nResponses: {'\n'.join(t['responses'])}" for t in crawl_data["queue"]["topics"]["head_topics"]])
    with open(os.path.join(RESULT_DIR, f"head_responses_{run_title}__{run_path}.txt"), "w") as f:
        f.write(head_responses_str)

    head_refusal_responses_str = "\n\n##########################################\n\n".join([f"Topic ID: {t['id']}\nRaw Query: {t['raw']}\nTranslation: {t['english']}\nResponses: {'\n'.join(t['responses'])}" for t in crawl_data["queue"]["topics"]["head_refusal_topics"]])
    with open(os.path.join(RESULT_DIR, f"head_refusal_responses_{run_title}__{run_path}.txt"), "w") as f:
        f.write(head_refusal_responses_str)

def make_match_comparison_table(crawl_names: List[CrawlName], gt_fnames: Dict[str, str], llm_judge_name: str, debug: bool = False, force_recompute: bool = False):
    # Load intermediate results if they exist
    all_titles_str = "--".join([name.title for name in crawl_names])
    result_fname = os.path.join(RESULT_DIR, f"crawl_matched_ranking_with_residuals_{all_titles_str}.json")
    if not os.path.exists(result_fname) or force_recompute:
        # Load crawl matched ranking
        all_clusters = {}
        for name in crawl_names:
            if name.acronym in all_clusters:
                raise ValueError(f"Crawl {name.acronym} already in all_clusters. We need unique acronyms.")
            all_clusters[name.acronym] = json.load(open(os.path.join(RESULT_DIR, f"topics_clustered_ranked_matched_{name.title}.json")))

        # get cluster residuals indexed as "acronym-cluster_id" --> "residual"
        all_unmatched_clusters = {}
        for acronym, clusters in all_clusters.items():
            for cluster_str, cluster_data in clusters.items():
                if not cluster_data["is_match"]:
                    all_unmatched_clusters[cluster_str] = [f"{acronym}:{cluster_str}"]

        
        clustered_unmatched_topics = llm_judge_topic_deduplication_batched(all_unmatched_clusters, all_titles_str, llm_judge_name, debug=debug, save_results=False, force_recompute=True)
        for residual_cluster_str, residual_cluster_ids in clustered_unmatched_topics.items():
            for residual_cluster_id in residual_cluster_ids:
                print(f"residual_cluster_id: {residual_cluster_id}")
                acronym, cluster_str = residual_cluster_id.split(":")
                all_clusters[acronym][cluster_str]["ground_truth_matches"] = [f'residual:residual:{residual_cluster_str}']
        
        with open(os.path.join(RESULT_DIR, f"crawl_matched_ranking_with_residuals_{all_titles_str}.json"), "w") as f:
            json.dump(all_clusters, f, indent=2)
    else:
        all_clusters = json.load(open(result_fname))
        clustered_unmatched_topics = set()
        # Iterate through acronyms first, then clusters
        for acronym, clusters in all_clusters.items():
            for cluster_str, cluster_data in clusters.items():
                if not cluster_data["is_match"]:
                    # Ensure ground_truth_matches exists before trying to access it
                    if "ground_truth_matches" in cluster_data:
                        residual_cluster_strs = [name.split(":")[-1] for name in cluster_data["ground_truth_matches"]]
                        clustered_unmatched_topics.update(residual_cluster_strs)
                    else:
                        # Handle cases where a non-matched cluster might not have ground_truth_matches yet
                        # This might indicate an issue with the data generation in the 'if' block or the loaded file
                        print(f"Warning: Cluster {acronym}:{cluster_str} is not matched but lacks 'ground_truth_matches'.")


    # For each crawl, load the aggregated topics, match them with the ground truth topics, and compute the residuals
    # Collect all ground truth topics gt_fname --> gt_category --> gt_topics --> plot_label : count
    topic_to_crawl_dict = {}
    for gt_dataset, gt_fdir in gt_fnames.items():
        gt_file = json.load(open(os.path.join(INPUT_DIR, f"{gt_fdir}.json")))
        for gt_category, gt_topics_list in gt_file.items():
            for gt_topic in gt_topics_list:
                gt_topic_string = f"{gt_dataset}:{gt_category}:{gt_topic}"
                topic_to_crawl_dict[gt_topic_string] = {
                    "category": gt_category,
                    "dataset": gt_dataset,
                }
    for unmatched_cluster in clustered_unmatched_topics:
        gt_topic_string = f"residual:residual:{unmatched_cluster}"
        topic_to_crawl_dict[gt_topic_string] = {
            "category": "residual",
            "dataset": "residual"
        }

    for name in crawl_names:
        for acronym, clusters in all_clusters.items():
            for cluster_str, cluster_data in clusters.items():
                for matched_gt_topic in cluster_data["ground_truth_matches"]:
                    if name.plot_label not in topic_to_crawl_dict[matched_gt_topic]:
                        topic_to_crawl_dict[matched_gt_topic][name.plot_label] = 0
                    topic_to_crawl_dict[matched_gt_topic][name.plot_label] += 1

    # Convert into a pd.DataFrame where each row is a key, and the columns are the keys of the inner dict
    df = pd.DataFrame.from_dict(topic_to_crawl_dict, orient='index')
    df.index.name = 'topic'  # Name the index column
    df = df.reset_index()  # Make topic a regular column
    
    format_topic_df_to_longtable(df)



# # "0328-tulu-8b-usersuffix-q0": "crawler_log_20250327_122827_Llama-3.1-Tulu-3-8B-SFT_1samples_100crawls_Truefilter_user_suffixprompt_q0.json",

if __name__ == "__main__":

    # Define all possible crawls
    all_crawl_names = [
        CrawlName(
            title="0407-claude-haiku-thought-prefix",
            path="crawler_log_20250407_000138_claude-3-5-haiku-latest_1samples_100000crawls_Truefilter_thought_prefixprompt_q8.json",
            acronym="C",
            plot_label="Claude-Haiku-3.5",
            model_name="claude-3-5-haiku-latest"
        ),
        CrawlName(
            title="0329-deepseek-70b-thought-prefix-q8",
            path="crawler_log_20250407_182051_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter_thought_prefixprompt_q8.json",
            acronym="D",
            plot_label="DeepSeek-70B",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        ),
        CrawlName(
            title="0329-meta-70b-assistant-prefix-q8",
            path="crawler_log_20250407_182048_Llama-3.3-70B-Instruct_1samples_100000crawls_Truefilter_assistant_prefixprompt_q8.json",
            acronym="M",
            plot_label="Meta-70B",
            model_name="meta-llama/Llama-3.3-70B-Instruct"
        ),
        CrawlName(
            title="0329-perplexity-70b-thought-prefix-q8",
            path="crawler_log_20250407_182048_r1-1776-distill-llama-70b_1samples_100000crawls_Truefilter_thought_prefixprompt_q8.json",
            acronym="P",
            plot_label="Perplexity-70B",
            model_name="perplexity-ai/r1-1776-distill-llama-70b"
        ),
        CrawlName(
            title="0328-tulu-8b-direct-prompting-q0",
            path="crawler_log_20250327_122827_Llama-3.1-Tulu-3-8B-SFT_1samples_100crawls_Truefilter_user_allprompt_q0.json",
            acronym="D",
            plot_label="Direct-Prompting",
            model_name="allenai/Llama-3.1-Tulu-3-8B-SFT"
        ),
        CrawlName(
            title="0328-tulu-8b-usersuffix-q0",
            path="crawler_log_20250327_122827_Llama-3.1-Tulu-3-8B-SFT_1samples_100crawls_Truefilter_user_suffixprompt_q0.json",
            acronym="U",
            plot_label="User-Suffix",
            model_name="allenai/Llama-3.1-Tulu-3-8B-SFT"
        ),
        CrawlName(
            title="0328-tulu-8b-assistant-prefix-q0",
            path="crawler_log_20250327_122827_Llama-3.1-Tulu-3-8B-SFT_1samples_100crawls_Truefilter_assistant_prefixprompt_q0.json",
            acronym="A",
            plot_label="Assistant-Prefix",
            model_name="allenai/Llama-3.1-Tulu-3-8B-SFT"
        ),
        CrawlName(
            title="0429-tulu-70b-assistant-prefix-q0",
            path="crawler_log_20250429_104903_Llama-3.1-Tulu-3-8B-SFT_1samples_100000crawls_Truefilter_assistant_prefixprompt_q0.json",
            acronym="A",
            plot_label="Assistant-Prefix",
            model_name="allenai/Llama-3.1-Tulu-3-8B-SFT"
        ),
        CrawlName(
            title="0430-tulu-70b-assistant-prefix-q0",
            path="crawler_log_20250430_124818_Llama-3.1-Tulu-3-8B-SFT_1samples_100000crawls_Truefilter_assistant_prefixprompt_q0.json",
            acronym="A",
            plot_label="Assistant-Prefix",
            model_name="allenai/Llama-3.1-Tulu-3-8B-SFT"
        ),
        CrawlName(
            title="0509-tulu-70b-assistant-prefix-q0",
            path="crawler_log_20250430_124818_Llama-3.1-Tulu-3-8B-SFT_1samples_100000crawls_Truefilter_assistant_prefixprompt_q0.json",
            acronym="A2",
            plot_label="Assistant-Prefix",
            model_name="allenai/Llama-3.1-Tulu-3-8B-SFT"
        ),
        CrawlName(
            title="0508-claude-haiku-thought-prefix",
            path="crawler_log_20250508_122246_claude-3-5-haiku-latest_1samples_1000crawls_Truefilter_thought_prefixprompt_q8.json",
            acronym="C",
            plot_label="Claude-Haiku-3.5",
            model_name="claude-3-5-haiku-latest"
        ),
        CrawlName(
            title="0206-gemma3-12b-thought-prefix-qNone",
            path="crawler_log_20250602_032140_gemma-3-12b-it_1samples_1000crawls_Truefilter_assistant_prefixprompt_qNone.json",
            acronym="G",
            plot_label="Gemma3-12B",
            model_name="gemma-3-12b-it"
        ),
        CrawlName(
            title="0603-mistral-thought-prefix-q8",
            path="crawler_log_20250603_035857_Mistral-Small-24B-Instruct-2501_1samples_1000crawls_Truefilter_thought_prefixprompt_q8.json",
            acronym="M",
            plot_label="Mistral-24B",
            model_name="mistralai/Mistral-Small-24B-Instruct-2501"
        ),
        CrawlName(
            title="deepseek-r1-0605-llama-8b",
            path="crawler_log_20250605_030457_DeepSeek-R1-Distill-Llama-8B_1samples_1000crawls_Truefilter_thought_prefixprompt_qNone.json",
            acronym="D",
            plot_label="DeepSeek-R1-0605-Llama-8B",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        ),
    ]
    all_crawl_names_dict = {name.title: name for name in all_crawl_names}

    available_analysis_modes = [
        'full', 
        'list_only',
        'rank_only', 
        'match_only', 
        'cluster_only', 
        'across_runs_only', 
        'summary_across_runs_only', 
        'convergence_across_runs_only'
        'rank_match_ROC'
    ]
    available_ranking_modes = [
        'clustered', 
        'individual'
    ]

    parser = argparse.ArgumentParser(description="Evaluate crawler results.")
    parser.add_argument("--crawl_titles", nargs='+', default=None, help="Subset of crawl titles to evaluate. If None, evaluates all defined crawls.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for caching models.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recomputation of intermediate results.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation (e.g., 'cuda', 'cpu').")
    parser.add_argument("--analysis_mode", type=str, default="full", choices=available_analysis_modes, help="Specify which parts of the analysis to run.")
    parser.add_argument("--llm_judge_name", type=str, default="gpt-4o", help="Name of the LLM judge model.")
    parser.add_argument("--ranking_mode", type=str, default="clustered", choices=available_ranking_modes, help="Whether to rank clustered or individual topics.")

    args = parser.parse_args()

    #########################################
    # Manual arg override
    args.analysis_mode = "list_only"
    args.ranking_mode = "individual"
    args.force_recompute = True
    args.debug = False
    args.crawl_titles = [
        # "0329-perplexity-70b-thought-prefix-q8", 
        # "0329-meta-70b-assistant-prefix-q8",
        # "0329-deepseek-70b-thought-prefix-q8",
        # "0407-claude-haiku-thought-prefix",

        # "0328-tulu-8b-direct-prompting-q0",
        # "0328-tulu-8b-usersuffix-q0",
        # "0328-tulu-8b-assistant-prefix-q0",

        # "0430-tulu-70b-assistant-prefix-q0"
        # "0509-tulu-70b-assistant-prefix-q0"
        # "0508-claude-haiku-thought-prefix"
        # "0206-gemma3-12b-thought-prefix-qNone"
        # "0603-mistral-thought-prefix-q8"
        "deepseek-r1-0605-llama-8b"
    ]

    gt_fnames = {
        "Tulu Safety": "tulu3_ground_truth_safety_topics",
        # "CCP censorship": "censorship_topics"
    }
    #########################################



    # Select crawls based on input or use all
    if args.crawl_titles:
        selected_crawl_names = [all_crawl_names_dict[title] for title in args.crawl_titles if title in all_crawl_names_dict]
        if len(selected_crawl_names) != len(args.crawl_titles):
            print("Warning: Some requested crawl titles were not found in the predefined list.")
    else:
        selected_crawl_names = all_crawl_names

    # Check for unique acronyms in the selected crawls
    check_unique_acronyms(selected_crawl_names)

    # Update variables from args
    CACHE_DIR = args.cache_dir
    debug = args.debug
    force_recompute = args.force_recompute
    DEVICE = args.device
    llm_judge_name = args.llm_judge_name
    analysis_mode = args.analysis_mode

    print(f"Running analysis with mode: {analysis_mode}")
    print(f"Selected crawls: {[name.title for name in selected_crawl_names]}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Debug: {debug}")
    print(f"Force recompute: {force_recompute}")
    print(f"LLM Judge: {llm_judge_name}")



    for names in selected_crawl_names:
        print(f"===== Processing: {names.title} =====")
        ## Load crawl data
        crawl_data = load_crawl(names.path)
        crawl_stats = CrawlerStats.load(crawl_data["stats"])
        print(f'num_steps: {len(crawl_stats.all_per_step)}')


        if analysis_mode in ['full', 'list_only']:
            ## Basic Stats
            print("Running basic stats...")
            print_stats(names.title, crawl_data, num_printed_refusals=10)
            write_head_refusal_topics(names.title, names.path, crawl_data)
            crawl_stats.visualize_cumulative_topic_count(save_path=os.path.join(RESULT_DIR, f"cumulative_topic_count_{names.title}__{names.path}.png"))

        if analysis_mode in ['full', 'cluster_only']:
            ## LLM Judge Topic Clustering
            print("Running LLM topic clustering...")
            llm_judge_aggregate_topics(names.title, crawl_data, llm_judge_name, debug=debug, force_recompute=force_recompute) # {cluster_str: {first_occurence_id, crawl_topics}}

        if analysis_mode in ['full', 'rank_only', 'rank_match_ROC']:
            print("Running self-ranking...")
            rank_topics(
                run_title=names.title,
                model_name=names.model_name,
                device=DEVICE,
                cache_dir=CACHE_DIR,
                ranking_mode=args.ranking_mode,
                crawl_data=crawl_data,
                force_recompute=force_recompute,
                debug=debug,
                config=None
            )
        
        if analysis_mode in ['full', 'rank_only', "wordcloud_only"]:
            print("Generating wordcloud...")
            generate_wordcloud_from_ranking(run_title=names.title, colormap="winter")

        if analysis_mode in ['full', 'match_only', 'rank_match_ROC']:
            ## Ground truth evaluation
            print("Running ground truth matching...")
            match_crawled_topics_with_gt(
                run_title=names.title,
                gt_topics_files=gt_fnames,
                llm_judge_name=llm_judge_name,
                verbose=True,
                force_recompute=force_recompute,
                debug=debug,
                ranking_mode=args.ranking_mode
            )
            # # Plot precision-recall curve
            # print("Plotting precision-recall curve...")
            # plot_precision_recall_curve(run_title=names.title, save_fig=True)

    # # Process across runs if requested and applicable
    if analysis_mode in ['full', 'across_runs_only', 'summary_across_runs_only']:
        print("===== Running cross-run analyses =====")
        print("Making match comparison table...")
        make_match_comparison_table(selected_crawl_names, gt_fnames, llm_judge_name, debug=debug, force_recompute=force_recompute)
        # Make joint convergence plot

    if analysis_mode in ['full', 'across_runs_only', 'convergence_across_runs_only']:
        print("Plotting joint convergence...")
        plot_first_occurrence_ids_across_runs(selected_crawl_names, RESULT_DIR)

    if analysis_mode in ['full', 'across_runs_only', 'ROC_across_runs_only', "rank_match_ROC"]:
        print("Plotting ROC curve...")
        plot_ROC_curve(selected_crawl_names, RESULT_DIR, ranking_mode=args.ranking_mode)

    print("===== Analysis complete =====")
            