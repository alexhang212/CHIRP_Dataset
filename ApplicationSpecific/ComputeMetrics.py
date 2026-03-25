"""Simplified script to compute metrics only - no inference components with optimizations"""
import os
import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import glob
import re
import warnings
warnings.filterwarnings('ignore')

# Global cache for loaded data
_data_cache = {}


ValidationVideos =['20210909_OiFHBq_1', '20210909_CFfo1Z', '20210914_h0h9CW',
                   '20210922_GWDswk', '20211018_WS3GWi', '20220816_RB15Aq', 
                   '20220829_BrmjDE', '20220901_WS3GWi', 
                   '20220922_ebqkf4', '20221004_ukRCut', '20221009_WS3GWi', 
                   '20221015_csG6Ba_2']

def load_data(file_path):
    """Load data from CSV or pickle file based on extension with caching"""
    if file_path in _data_cache:
        return _data_cache[file_path]
    
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.p') or file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    _data_cache[file_path] = data
    return data


def get_bbox_overlap(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes
    
    Args:
        box1, box2: [xmin, ymin, xmax, ymax]
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def discover_algorithm_combinations(inference_dir):
    """
    Discover all available algorithm combinations from the inference directory
    
    Returns:
        list: List of (model_name, id_algo, tracker) tuples
    """    
    # Patterns to extract algorithm names
    behav_pattern = r'.*_BehavDict_(.+)_(.+)\.p$'
    id_pattern = r'.*_(.+)_IDMatch_(.+)\.p$'
    tracker_pattern = r'.*_TrackedBBox_(.+)\.csv$'
    
    models = set()
    id_algos = set()
    trackers = set()
    
    # Scan all files in the directory
    for filename in os.listdir(inference_dir):
        # Extract from BehavDict files: {video}_BehavDict_{model}_{tracker}.p
        behav_match = re.match(behav_pattern, filename)
        if behav_match:
            model, tracker = behav_match.groups()
            models.add(model)
            trackers.add(tracker)
        
        # Extract from IDMatch files: {video}_{id_algo}_IDMatch_{tracker}.p
        id_match = re.match(id_pattern, filename)
        if id_match:
            id_algo, tracker = id_match.groups()
            id_algos.add(id_algo)
            trackers.add(tracker)
        
        # Extract from TrackedBBox files: {video}_TrackedBBox_{tracker}.csv
        tracker_match = re.match(tracker_pattern, filename)
        if tracker_match:
            tracker = tracker_match.groups()[0]
            trackers.add(tracker)
    
    # Generate all possible combinations
    combinations = []
    for model in models:
        for id_algo in id_algos:
            for tracker in trackers:
                combinations.append((model, id_algo, tracker))
    
    print(f"Discovered {len(models)} models: {sorted(models)}")
    print(f"Discovered {len(id_algos)} ID algorithms: {sorted(id_algos)}")
    print(f"Discovered {len(trackers)} trackers: {sorted(trackers)}")
    print(f"Total combinations to process: {len(combinations)}")
    
    return combinations


def find_inference_files(inference_dir, model_name, id_algo, tracker):
    """
    Find inference files based on naming convention
    
    Expected naming convention:
    - BehavDict: {video}_BehavDict_{model}_{tracker}.p
    - TrackedBBox: {video}_TrackedBBox_{tracker}.csv  
    - IDMatch: {video}_{id_algo}_IDMatch_{tracker}.p
    
    Returns:
        dict: {video_name: {'behav_dict': path, 'tracked_bbox': path, 'id_match': path}}
    """
    
    found_files = {}
    
    for video in ValidationVideos:
        video_files = {}
        
        # Look for BehavDict file
        behav_pattern = os.path.join(inference_dir, f"{video}_BehavDict_{model_name}_{tracker}.p")
        behav_files = glob.glob(behav_pattern)
        if behav_files:
            video_files['behav_dict'] = behav_files[0]
        else:
            continue
            
        # Look for TrackedBBox file
        bbox_pattern = os.path.join(inference_dir, f"{video}_TrackedBBox_{tracker}.csv")
        bbox_files = glob.glob(bbox_pattern)
        if bbox_files:
            video_files['tracked_bbox'] = bbox_files[0]
        else:
            continue
            
        # Look for IDMatch file
        id_pattern = os.path.join(inference_dir, f"{video}_{id_algo}_IDMatch_{tracker}.p")
        id_files = glob.glob(id_pattern)
        if id_files:
            video_files['id_match'] = id_files[0]
        else:
            continue
            
        if len(video_files) == 3:  # All files found
            found_files[video] = video_files
    
    return found_files


def check_results_exist(output_dir, model_name, id_algo, tracker):
    """
    Check if results already exist for this combination
    
    Returns:
        bool: True if results exist, False otherwise
    """
    output_prefix = f"{id_algo}_{tracker}_{model_name}"
    metrics_file = os.path.join(output_dir, f"{output_prefix}_metrics.pkl")
    return os.path.exists(metrics_file)


def create_summary_csv(output_dir, human_benchmark_data=None, random_benchmark_data=None):
    """
    Read all output files and create a comprehensive summary CSV
    
    Args:
        output_dir: Directory containing all the output files
        human_benchmark_data: Optional tuple of (PeckRateDicts, PropTimeDicts, BehaviorDicts, FrameAccuracy)
        random_benchmark_data: Optional tuple of (PeckRateDicts, PropTimeDicts, BehaviorDicts, FrameAccuracy)
    """
    print("\nCreating comprehensive summary CSV...")
    
    # Find all metrics files
    metrics_files = glob.glob(os.path.join(output_dir, "*_metrics.pkl"))
    
    if not metrics_files and not human_benchmark_data:
        print("No metrics files found. Cannot create summary.")
        return
    
    print(f"Found {len(metrics_files)} result sets to summarize")
    
    # Extract algorithm info from filenames
    def parse_filename(filepath):
        basename = os.path.basename(filepath)
        # Pattern: {id_algo}_{tracker}_{model}_*.{ext}
        parts = basename.split('_')
        if len(parts) >= 3:
            id_algo = parts[0]
            tracker = parts[1]
            model = parts[2]
            return id_algo, tracker, model
        return None, None, None
    
    summary_data = []
    
    for metrics_file in metrics_files:
        id_algo, tracker, model = parse_filename(metrics_file)
        if not all([id_algo, tracker, model]):
            continue
            
        try:
            # Load summary metrics (frame accuracy)
            summary_file = metrics_file.replace("_metrics.pkl", "_summary.pkl")
            if os.path.exists(summary_file):
                with open(summary_file, 'rb') as f:
                    summary_metrics = pickle.load(f)
                frame_correct = list(summary_metrics["PerFramePercentageCorrect"].values())[0]
                track_correct = list(summary_metrics["TrackPercentageCorrect"].values())[0] if "TrackPercentageCorrect" in summary_metrics else np.nan
            else:
                frame_correct = np.nan
                track_correct = np.nan
            
            # Load peck rate metrics
            peck_rate_file = metrics_file.replace("_metrics.pkl", "_peck_rates.csv")
            if os.path.exists(peck_rate_file):
                peck_df = pd.read_csv(peck_rate_file)
                peck_df["AbsError"] = (peck_df["GTFeedRate"] - peck_df["PredFeedRate"]).abs()
                mean_peck_error = peck_df["AbsError"].mean()
                median_peck_error = peck_df["AbsError"].median()
                std_peck_error = peck_df["AbsError"].std()
                peck_correlation = peck_df["GTFeedRate"].corr(peck_df["PredFeedRate"])
                total_peck_measurements = len(peck_df)
            else:
                mean_peck_error = median_peck_error = std_peck_error = peck_correlation = np.nan
                total_peck_measurements = 0
            
            # Load PropTime metrics
            proptime_file = metrics_file.replace("_metrics.pkl", "_proptime.csv")
            if os.path.exists(proptime_file):
                proptime_df = pd.read_csv(proptime_file)
                proptime_df["AbsError"] = (proptime_df["GT_PropTime"] - proptime_df["Pred_PropTime"]).abs()
                mean_proptime_error = proptime_df["AbsError"].mean()
                median_proptime_error = proptime_df["AbsError"].median()
                std_proptime_error = proptime_df["AbsError"].std()
                proptime_correlation = proptime_df["GT_PropTime"].corr(proptime_df["Pred_PropTime"])
                total_proptime_measurements = len(proptime_df)
            else:
                mean_proptime_error = median_proptime_error = std_proptime_error = proptime_correlation = np.nan
                total_proptime_measurements = 0
            
            # Load precision/recall metrics
            pr_file = metrics_file.replace("_metrics.pkl", "_precision_recall.csv")
            if os.path.exists(pr_file):
                pr_df = pd.read_csv(pr_file)
                mean_precision = pr_df["Peck_Precision"].mean()
                mean_recall = pr_df["Peck_Recall"].mean()
                mean_f1 = pr_df["Peck_F1"].mean()
                total_behavior_measurements = len(pr_df)
            else:
                mean_precision = mean_recall = mean_f1 = np.nan
                total_behavior_measurements = 0
            
            # Compile summary row
            summary_row = {
                "ID_Algorithm": id_algo,
                "Tracker": tracker,
                "Behavior_Model": model,
                "Combination": f"{id_algo}_{tracker}_{model}",
                
                # Frame accuracy metrics
                "PerFrame_Percentage_Correct": frame_correct,
                "Track_Percentage_Correct": track_correct,

                # Feed rate metrics
                "Mean_FeedRate_AbsError": mean_peck_error,
                "Median_FeedRate_AbsError": median_peck_error,
                "Std_FeedRate_AbsError": std_peck_error,
                "FeedRate_Correlation": peck_correlation,
                "Total_FeedRate_Measurements": total_peck_measurements,

                # Social interaction (PropTime) metrics
                "Mean_PropTime_AbsError": mean_proptime_error,
                "Median_PropTime_AbsError": median_proptime_error,
                "Std_PropTime_AbsError": std_proptime_error,
                "PropTime_Correlation": proptime_correlation,
                "Total_PropTime_Measurements": total_proptime_measurements,

                # Behavior classification metrics
                "Mean_Peck_Precision": mean_precision,
                "Mean_Peck_Recall": mean_recall,
                "Mean_Peck_F1": mean_f1,
                "Total_Behavior_Measurements": total_behavior_measurements
            }

            summary_data.append(summary_row)
            
        except Exception as e:
            print(f"Error processing {metrics_file}: {str(e)}")
            continue
    
    # Add human benchmark data if provided
    if human_benchmark_data:
        peck_rate_dicts, proptime_dicts, behavior_dicts, frame_accuracy = human_benchmark_data
        
        # Process human peck rate data
        peck_df = pd.DataFrame(peck_rate_dicts)
        if not peck_df.empty:
            peck_df["AbsError"] = (peck_df["GTFeedRate"] - peck_df["PredFeedRate"]).abs()
            mean_peck_error = peck_df["AbsError"].mean()
            median_peck_error = peck_df["AbsError"].median()
            std_peck_error = peck_df["AbsError"].std()
            peck_correlation = peck_df["GTFeedRate"].corr(peck_df["PredFeedRate"])
            total_peck_measurements = len(peck_df)
        else:
            mean_peck_error = median_peck_error = std_peck_error = peck_correlation = np.nan
            total_peck_measurements = 0
        
        # Process human PropTime data
        proptime_df = pd.DataFrame(proptime_dicts)
        if not proptime_df.empty:
            proptime_df["AbsError"] = (proptime_df["GT_PropTime"] - proptime_df["Pred_PropTime"]).abs()
            mean_proptime_error = proptime_df["AbsError"].mean()
            median_proptime_error = proptime_df["AbsError"].median()
            std_proptime_error = proptime_df["AbsError"].std()
            proptime_correlation = proptime_df["GT_PropTime"].corr(proptime_df["Pred_PropTime"])
            total_proptime_measurements = len(proptime_df)
        else:
            mean_proptime_error = median_proptime_error = std_proptime_error = proptime_correlation = np.nan
            total_proptime_measurements = 0
        
        # Process human behavior data (set default values since GetHumanBench doesn't compute these)
        mean_precision = mean_recall = mean_f1 = np.nan
        total_behavior_measurements = 0
        
        # Human benchmark summary row
        human_summary = {
            "ID_Algorithm": "Human",
            "Tracker": "Human", 
            "Behavior_Model": "Human",
            "Combination": "Human_Human_Human",
            
            # Frame accuracy metrics
            "PerFrame_Percentage_Correct": frame_accuracy,
            "Track_Percentage_Correct": np.nan,

            # Feed rate metrics
            "Mean_FeedRate_AbsError": mean_peck_error,
            "Median_FeedRate_AbsError": median_peck_error,
            "Std_FeedRate_AbsError": std_peck_error,
            "FeedRate_Correlation": peck_correlation,
            "Total_FeedRate_Measurements": total_peck_measurements,

            # Social interaction (PropTime) metrics
            "Mean_PropTime_AbsError": mean_proptime_error,
            "Median_PropTime_AbsError": median_proptime_error,
            "Std_PropTime_AbsError": std_proptime_error,
            "PropTime_Correlation": proptime_correlation,
            "Total_PropTime_Measurements": total_proptime_measurements,

            # Behavior classification metrics (not available for human benchmark)
            "Mean_Peck_Precision": mean_precision,
            "Mean_Peck_Recall": mean_recall,
            "Mean_Peck_F1": mean_f1,
            "Total_Behavior_Measurements": total_behavior_measurements
        }

        summary_data.append(human_summary)
        print("✅ Added human benchmark data to summary")
    
    # Add random benchmark data if provided (multiple combinations)
    if random_benchmark_data:
        for combination_name, combo_data in random_benchmark_data.items():
            # Process peck rate data for this combination
            peck_data = combo_data['peck_data']
            if peck_data:
                peck_df = pd.DataFrame(peck_data)
                # Use AbsError column if available, otherwise calculate it
                if 'AbsError' in peck_df.columns:
                    peck_df["AbsError"] = peck_df["AbsError"]
                else:
                    peck_df["AbsError"] = (peck_df["GTFeedRate"] - peck_df["PredFeedRate"]).abs()
                
                mean_peck_error = peck_df["AbsError"].mean()
                median_peck_error = peck_df["AbsError"].median()
                std_peck_error = peck_df["AbsError"].std()
                peck_correlation = peck_df["GTFeedRate"].corr(peck_df["PredFeedRate"])
                total_peck_measurements = len(peck_df)
            else:
                mean_peck_error = median_peck_error = std_peck_error = peck_correlation = np.nan
                total_peck_measurements = 0
            
            # Process PropTime data for this combination
            proptime_data = combo_data['proptime_data']
            if proptime_data:
                proptime_df = pd.DataFrame(proptime_data)
                # Use existing AbsError or calculate from GT_PropTime and Pred_PropTime columns
                if 'AbsError' in proptime_df.columns:
                    proptime_df["AbsError"] = proptime_df["AbsError"]
                else:
                    proptime_df["AbsError"] = (proptime_df["GT_PropTime"] - proptime_df["Pred_PropTime"]).abs()
                
                mean_proptime_error = proptime_df["AbsError"].mean()
                median_proptime_error = proptime_df["AbsError"].median()
                std_proptime_error = proptime_df["AbsError"].std()
                proptime_correlation = proptime_df["GT_PropTime"].corr(proptime_df["Pred_PropTime"])
                total_proptime_measurements = len(proptime_df)
            else:
                mean_proptime_error = median_proptime_error = std_proptime_error = proptime_correlation = np.nan
                total_proptime_measurements = 0
            
            # Process behavior data for this combination
            behavior_data = combo_data['behavior_data']
            if behavior_data:
                behavior_df = pd.DataFrame(behavior_data)
                mean_precision = behavior_df["Peck_Precision"].mean()
                mean_recall = behavior_df["Peck_Recall"].mean()
                mean_f1 = behavior_df["Peck_F1"].mean()
                total_behavior_measurements = len(behavior_df)
            else:
                mean_precision = mean_recall = mean_f1 = np.nan
                total_behavior_measurements = 0
            
            # Create summary row for this random combination
            random_summary = {
                "ID_Algorithm": combo_data['id_algo'],
                "Tracker": combo_data['tracker'], 
                "Behavior_Model": combo_data['model'],
                "Combination": combination_name,
                
                # Frame accuracy metrics
                "PerFrame_Percentage_Correct": combo_data['frame_accuracy'],
                "Track_Percentage_Correct": np.nan,

                # Feed rate metrics
                "Mean_FeedRate_AbsError": mean_peck_error,
                "Median_FeedRate_AbsError": median_peck_error,
                "Std_FeedRate_AbsError": std_peck_error,
                "FeedRate_Correlation": peck_correlation,
                "Total_FeedRate_Measurements": total_peck_measurements,
                
                # Social interaction (PropTime) metrics
                "Mean_PropTime_AbsError": mean_proptime_error,
                "Median_PropTime_AbsError": median_proptime_error,
                "Std_PropTime_AbsError": std_proptime_error,
                "PropTime_Correlation": proptime_correlation,
                "Total_PropTime_Measurements": total_proptime_measurements,
                
                # Behavior classification metrics
                "Mean_Peck_Precision": mean_precision,
                "Mean_Peck_Recall": mean_recall,
                "Mean_Peck_F1": mean_f1,
                "Total_Behavior_Measurements": total_behavior_measurements
            }
            
            summary_data.append(random_summary)
        
        print(f"✅ Added {len(random_benchmark_data)} random benchmark combinations to summary")
    
    if not summary_data:
        print("No valid data found for summary.")
        return
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by performance (default sorting)
    summary_df = summary_df.sort_values(['PerFrame_Percentage_Correct', 'FeedRate_Correlation'], 
                                       ascending=[False, False], na_position='last')
    
    # Save comprehensive summary
    summary_csv_path = os.path.join(output_dir, "comprehensive_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Create a simplified ranking summary
    ranking_df = summary_df[['Combination', 'ID_Algorithm', 'Tracker', 'Behavior_Model',
                            'PerFrame_Percentage_Correct', 'Track_Percentage_Correct',
                            'Mean_FeedRate_AbsError', 'FeedRate_Correlation',
                            'Mean_PropTime_AbsError', 'PropTime_Correlation',
                            'Mean_Peck_Precision', 'Mean_Peck_Recall', 'Mean_Peck_F1']].copy()
    
    ranking_csv_path = os.path.join(output_dir, "ranking_summary.csv")
    ranking_df.to_csv(ranking_csv_path, index=False)
    
    print(f"✅ Comprehensive summary saved: {summary_csv_path}")
    print(f"✅ Ranking summary saved: {ranking_csv_path}")
    print(f"📊 Summarized {len(summary_df)} algorithm combinations")
    
    # Filter out benchmarks for rankings (but keep in full summary)
    # Human benchmark has Combination = 'Human_Human_Human'
    # Random benchmarks have ID_Algorithm = 'Random'
    algorithm_df = summary_df[
        (summary_df['Combination'] != 'Human_Human_Human') & 
        (summary_df['ID_Algorithm'] != 'Random')
    ].copy()
    
    if len(algorithm_df) > 0:
        # Create feed rate ranking (low error + high correlation) - algorithms only
        print("\n🥇 Top 5 Feed Rate Performers (low error + high correlation):")
        feedrate_ranking = algorithm_df.copy()
        feedrate_ranking = feedrate_ranking.sort_values(['Mean_FeedRate_AbsError', 'FeedRate_Correlation'], 
                                                       ascending=[True, False], na_position='last')
        
        top_5_feedrate = feedrate_ranking.head(5)
        for i, (_, row) in enumerate(top_5_feedrate.iterrows(), 1):
            print(f"{i}. {row['Combination']}: {row['Mean_FeedRate_AbsError']:.3f} error, "
                  f"{row['FeedRate_Correlation']:.3f} correlation")
        
        # Create PropTime ranking (low error + high correlation) - algorithms only
        print("\n🥈 Top 5 PropTime Performers (low error + high correlation):")
        proptime_ranking = algorithm_df.copy()
        proptime_ranking = proptime_ranking.sort_values(['Mean_PropTime_AbsError', 'PropTime_Correlation'], 
                                                       ascending=[True, False], na_position='last')
        
        top_5_proptime = proptime_ranking.head(5)
        for i, (_, row) in enumerate(top_5_proptime.iterrows(), 1):
            print(f"{i}. {row['Combination']}: {row['Mean_PropTime_AbsError']:.3f} error, "
                  f"{row['PropTime_Correlation']:.3f} correlation")
        
        # Show benchmark performances for comparison
        print(f"\n📊 Benchmark Comparisons:")
        
        # Human benchmark
        human_row = summary_df[summary_df['Combination'] == 'Human_Human_Human']
        if not human_row.empty:
            human_data = human_row.iloc[0]
            print(f"   🧑 Human: Feed Rate {human_data['Mean_FeedRate_AbsError']:.3f} error, "
                  f"{human_data['FeedRate_Correlation']:.3f} corr | PropTime {human_data['Mean_PropTime_AbsError']:.3f} error, "
                  f"{human_data['PropTime_Correlation']:.3f} corr | Frame Acc {human_data['PerFrame_Percentage_Correct']:.3f}")
        
    else:
        print("\nNo algorithm results found for ranking.")


def expand_grid(dictionary):
    from itertools import product
    return pd.DataFrame([row for row in product(*dictionary.values())], 
                        columns=dictionary.keys())


def GetHumanBench(HumanDir, GTDataDir, ValidationVideos, Duration=5):
    """Get human benchmark data for validation videos, Duration is in minutes"""
    fps = 25
    MaxFrame = Duration * 60 * fps
    PeckRateDicts = []
    PropTimeDicts = []
    all_frame_accuracies = []
    
    for vid in ValidationVideos:
        GTBBox = load_data(os.path.join(GTDataDir, vid + "_BBox.csv"))
        GTBBox["RealID"] = GTBBox["ID"].apply(lambda x: x.split("_")[0])
        GTBehav = load_data(os.path.join(GTDataDir, vid + "_PeckEvents.csv"))

        GTBBox = GTBBox.loc[GTBBox["Frame"] < MaxFrame]
        GTBehav = GTBehav.loc[GTBehav["EndFrame"] < MaxFrame]
        GTBBox = GTBBox.loc[GTBBox["RealID"] != "unknown"]

        UnqIDs = GTBBox["RealID"].unique()

        AllBORIS = []
        
        # Collect frame accuracy data (assuming perfect frame accuracy for human benchmark)
        # Since humans manually annotated the ground truth, frame accuracy should be 1.0
        video_frame_accuracy = 1.0
        all_frame_accuracies.append(video_frame_accuracy)
        
        for ID in UnqIDs:
            IDBehav = GTBehav[(GTBehav["ID"] == ID) & (GTBehav["Behaviour"] == "peck")]
            GTFeedRate = len(IDBehav.index)/Duration

            try:
                BorisDF = pd.read_csv(os.path.join(HumanDir, "%s_%s.tsv"%(vid, ID)), sep="\t")
                BorisDF = BorisDF.iloc[:MaxFrame]
                BorisDF["Present_%s"%ID] = BorisDF["Present"] 
                
                AllBORIS.append(BorisDF)
                PredFeedRate = sum(BorisDF["Peck"])/Duration

                PeckRateDicts.append({"Video":vid,
                                      "ID":ID,
                                      "Model": "Human",
                                      "IDAlgo": "Human",
                                      "Tracker": "Human",
                                      "Name": "Human_Benchmark",
                                      "GTFeedRate":GTFeedRate,
                                      "PredFeedRate":PredFeedRate})
            except FileNotFoundError:
                print(f"Warning: Human benchmark file not found for {vid}_{ID}")
                continue
        
        if AllBORIS:  # Only process if we have BORIS data
            CombinedDF = pd.concat(AllBORIS, axis=1)

            AllIDPairs = list(itertools.combinations(GTBBox["RealID"].unique(), 2))

            for pair in AllIDPairs:
                Pair1Frames = GTBBox[GTBBox["RealID"] == pair[0]]["Frame"].tolist()
                Pair2Frames = GTBBox[GTBBox["RealID"] == pair[1]]["Frame"].tolist()

                GTOverlapFrames = set(Pair1Frames).intersection(set(Pair2Frames))
                GT_PropTime = len(GTOverlapFrames) / MaxFrame

                try:
                    BorisPairDF = CombinedDF[["Present_%s"%pair[0], "Present_%s"%pair[1]]]
                    Pair1Frames = BorisPairDF.loc[BorisPairDF["Present_%s"%pair[0]]==1].index.tolist()
                    Pair2Frames = BorisPairDF.loc[BorisPairDF["Present_%s"%pair[1]]==1].index.tolist()
                    
                    PredOverlapFrames = set(Pair1Frames).intersection(set(Pair2Frames))
                    Pred_PropTime = len(PredOverlapFrames) / MaxFrame
                    
                    PropTimeDicts.append({"Video":vid,
                                          "ID1":pair[0],
                                          "ID2":pair[1],
                                          "Model": "Human",
                                          "IDAlgo": "Human",
                                          "Tracker": "Human",
                                          "Name": "Human_Benchmark",
                                          "GT_PropTime":GT_PropTime,
                                          "Pred_PropTime":Pred_PropTime})
                except KeyError:
                    # Skip this pair if human data is missing
                    continue
    
    # Calculate overall frame accuracy (average across videos)
    overall_frame_accuracy = np.mean(all_frame_accuracies) if all_frame_accuracies else 1.0
    
    return PeckRateDicts, PropTimeDicts, [], overall_frame_accuracy


def GetRandomBench(HumanDir):
    """Get random benchmark data from CSV files and organize by combination"""
    try:
        # Load random benchmark CSV files
        peck_rate_file = os.path.join(HumanDir, "Random_PeckRateData.csv")
        pair_data_file = os.path.join(HumanDir, "Random_PairData.csv") 
        precision_recall_file = os.path.join(HumanDir, "Random_PrecRecallData.csv")
        frame_accuracy_file = os.path.join(HumanDir, "Random_PerFramePercentageCorrect.csv")
        
        # Check if files exist
        if not all(os.path.exists(f) for f in [peck_rate_file, pair_data_file, precision_recall_file, frame_accuracy_file]):
            return None
        
        # Load the data
        peck_df = pd.read_csv(peck_rate_file)
        pair_df = pd.read_csv(pair_data_file)
        precision_recall_df = pd.read_csv(precision_recall_file)
        frame_accuracy_df = pd.read_csv(frame_accuracy_file)
        
        # Organize data by combination (IDAlgo_Tracker_Model)
        combinations_data = {}
        
        # Process frame accuracy data (one per combination)
        for _, row in frame_accuracy_df.iterrows():
            combination = f"{row['IDAlgo']}_{row['Tracker']}_{row['Model']}"
            if combination not in combinations_data:
                combinations_data[combination] = {
                    'id_algo': row['IDAlgo'],
                    'tracker': row['Tracker'],
                    'model': row['Model'],
                    'frame_accuracy': row['PerFramePercentageCorrect'],
                    'peck_data': [],
                    'proptime_data': [],
                    'behavior_data': []
                }
        
        # Add peck rate data
        for _, row in peck_df.iterrows():
            combination = f"{row['IDAlgo']}_{row['Tracker']}_{row['Model']}"
            if combination in combinations_data:
                combinations_data[combination]['peck_data'].append(row.to_dict())
        
        # Add proptime data
        for _, row in pair_df.iterrows():
            combination = f"{row['IDAlgo']}_{row['Tracker']}_{row['Model']}"
            if combination in combinations_data:
                combinations_data[combination]['proptime_data'].append(row.to_dict())
        
        # Add behavior data
        for _, row in precision_recall_df.iterrows():
            combination = f"{row['IDAlgo']}_{row['Tracker']}_{row['Model']}"
            if combination in combinations_data:
                combinations_data[combination]['behavior_data'].append(row.to_dict())
        
        return combinations_data
        
    except Exception as e:
        print(f"Error loading random benchmark data: {str(e)}")
        return None


def get_bbox_overlap_vectorized(boxes1, boxes2):
    """Vectorized bounding box overlap calculation (intersection / area of boxes1)

    Args:
        boxes1: numpy array of shape (N, 4) with format [xmin, ymin, xmax, ymax]
        boxes2: numpy array of shape (M, 4) with format [xmin, ymin, xmax, ymax]

    Returns:
        numpy array of shape (N, M) with overlap values (intersection / area of box1)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.array([])

    boxes1 = boxes1.astype(np.float32)
    boxes2 = boxes2.astype(np.float32)

    # Expand dimensions for broadcasting
    boxes1 = np.expand_dims(boxes1, axis=1)  # (N, 1, 4)
    boxes2 = np.expand_dims(boxes2, axis=0)  # (1, M, 4)

    # Calculate intersection dimensions
    dx = np.minimum(boxes1[:, :, 2], boxes2[:, :, 2]) - np.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
    dy = np.minimum(boxes1[:, :, 3], boxes2[:, :, 3]) - np.maximum(boxes1[:, :, 1], boxes2[:, :, 1])

    # Intersection area (zero where no overlap)
    intersection = np.where((dx >= 0) & (dy >= 0), dx * dy, 0.0)

    # Area of boxes1
    area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])

    return np.divide(intersection, area1, out=np.zeros_like(intersection), where=area1 > 0)


def compute_feed_rates_optimized(GTBehav, BehavDict, IDMatch, VideoMinutes):
    """Optimized feed rate calculation"""
    # Vectorized GT counts
    gt_counts = GTBehav.groupby("ID").size()
    gt_rates = gt_counts / VideoMinutes
    
    # Efficient predicted peck counting
    pred_counts = {}
    for track, track_id in IDMatch.items():
        if track in BehavDict and track_id != "unringed":
            peck_count = sum(1 for val in BehavDict[track].values() if val == "Peck")
            pred_counts[track_id] = pred_counts.get(track_id, 0) + peck_count
    
    # Create results efficiently
    results = []
    for gt_id in gt_rates.index:
        gt_rate = gt_rates[gt_id]
        pred_rate = pred_counts.get(gt_id, 0) / VideoMinutes
        results.append({
            "ID": gt_id, "GTFeedRate": gt_rate, "PredFeedRate": pred_rate
        })
    
    return results


def compute_proptime_optimized(GTBBox, TrackedBBox, IDMatch, VideoLength):
    """Optimized PropTime calculation using set operations"""
    results = []
    
    # Prepare data efficiently
    GTBBox["RealID"] = GTBBox["ID"].str.split("_").str[0]
    GTBBox = GTBBox[GTBBox["RealID"] != "unknown"].copy()
    
    TrackedBBox["RealID"] = TrackedBBox["ID"].map(IDMatch).fillna(np.nan)
    TrackedBBox = TrackedBBox.dropna(subset=["RealID"]).copy()
    
    # Get frame sets once
    gt_frames_by_id = GTBBox.groupby("RealID")["Frame"].apply(set).to_dict()
    pred_frames_by_id = TrackedBBox.groupby("RealID")["Frame"].apply(set).to_dict()
    
    # Generate pairs and compute efficiently
    unique_ids = list(gt_frames_by_id.keys())
    for i in range(len(unique_ids)):
        for j in range(i + 1, len(unique_ids)):
            id1, id2 = unique_ids[i], unique_ids[j]
            
            # GT overlap
            gt_frames1 = gt_frames_by_id.get(id1, set())
            gt_frames2 = gt_frames_by_id.get(id2, set())
            gt_overlap = len(gt_frames1 & gt_frames2)
            gt_prop_time = gt_overlap / VideoLength
            
            # Pred overlap
            pred_frames1 = pred_frames_by_id.get(id1, set())
            pred_frames2 = pred_frames_by_id.get(id2, set())
            pred_overlap = len(pred_frames1 & pred_frames2)
            pred_prop_time = pred_overlap / VideoLength
            
            results.append({
                "ID1": id1, "ID2": id2,
                "GT_PropTime": gt_prop_time, "Pred_PropTime": pred_prop_time
            })
    
    return results


def compute_behavior_metrics_optimized(GTBehav, BehavDict, IDMatch, VideoLength):
    """Optimized behavior metrics with efficient set operations"""
    # Create GT frame sets efficiently
    gt_peck_frames = {}
    for _, row in GTBehav.iterrows():
        bird_id = row["ID"]
        if bird_id not in gt_peck_frames:
            gt_peck_frames[bird_id] = set()
        
        start_frame = int(row["StartFrame"])
        end_frame = int(row["EndFrame"])
        gt_peck_frames[bird_id].update(range(start_frame, end_frame + 1))
    
    # Create predicted frame sets efficiently
    pred_peck_frames = {}
    for track, track_id in IDMatch.items():
        if track in BehavDict and track_id != "unringed":
            if track_id not in pred_peck_frames:
                pred_peck_frames[track_id] = set()
            
            for (start_frame, end_frame), behavior in BehavDict[track].items():
                if behavior == "Peck":
                    pred_peck_frames[track_id].update(range(start_frame, end_frame + 1))
    
    # Compute precision/recall efficiently using vectorized time windows
    time_windows = np.arange(0, VideoLength, 25)
    results = []
    
    for bird_id in gt_peck_frames.keys():
        gt_frames = gt_peck_frames[bird_id]
        pred_frames = pred_peck_frames.get(bird_id, set())
        
        # Vectorized window computation
        gt_windows = np.array([
            any(frame in gt_frames for frame in range(start, min(start + 25, VideoLength)))
            for start in time_windows
        ], dtype=int)
        
        pred_windows = np.array([
            any(frame in pred_frames for frame in range(start, min(start + 25, VideoLength)))
            for start in time_windows
        ], dtype=int)
        
        if len(np.unique(gt_windows)) > 1:  # Only compute if there's variation
            precision = precision_score(gt_windows, pred_windows, average=None, zero_division=0)
            recall = recall_score(gt_windows, pred_windows, average=None, zero_division=0)
            f1 = f1_score(gt_windows, pred_windows, average=None, zero_division=0)
            
            # Get positive class metrics (index 1)
            prec_val = precision[1] if len(precision) > 1 else 0
            rec_val = recall[1] if len(recall) > 1 else 0
            f1_val = f1[1] if len(f1) > 1 else 0
            
            results.append({
                "ID": bird_id, "Peck_Precision": prec_val, 
                "Peck_Recall": rec_val, "Peck_F1": f1_val
            })
    
    return results


def compute_frame_accuracy_optimized(GTBBox, TrackedBBox, IDMatch):
    """Frame accuracy calculation returning per-frame and track-wise accuracy.

    Returns:
        per_frame_accuracy: fraction of frames where the correct bird is detected
        track_accuracy: fraction of tracks where >80% of frames are correct
    """
    # Prepare data
    GTBBox["RealID"] = GTBBox["ID"].str.split("_").str[0]
    GTBBox = GTBBox[GTBBox["RealID"] != "unknown"].copy()

    TrackedBBox["RealID"] = TrackedBBox["ID"].map(IDMatch).fillna(np.nan)
    # TrackedBBox = TrackedBBox.dropna(subset=["RealID"]).copy()

    pred_by_frame = TrackedBBox.groupby("Frame")

    OverallTally = []   # 1 per track: was >80% of its frames correct?
    PerFrameTally = []  # 1 per frame: was this frame correct?

    for track_id in GTBBox["ID"].unique():
        track_data = GTBBox[GTBBox["ID"] == track_id]
        gt_real_id = track_data["RealID"].iloc[0]

        TallyList = []

        for _, gt_row in track_data.iterrows():
            frame = gt_row["Frame"]
            gt_box = np.array([gt_row["Xmin"], gt_row["Ymin"], gt_row["Xmax"], gt_row["Ymax"]])

            if frame not in pred_by_frame.groups:
                TallyList.append(0)
                continue

            pred_frame_data = pred_by_frame.get_group(frame)
            pred_boxes = pred_frame_data[["Xmin", "Ymin", "Xmax", "Ymax"]].values
            pred_ids = pred_frame_data["RealID"].values

            if len(pred_boxes) == 0:
                TallyList.append(0)
                continue

            overlaps = get_bbox_overlap_vectorized(gt_box.reshape(1, -1), pred_boxes)[0]
            overlap_indices = np.where(overlaps > 0.5)[0]

            if len(overlap_indices) == 0:
                TallyList.append(0)
                continue

            pred_id = pred_ids[overlap_indices[0]]
            TallyList.append(1 if pred_id == gt_real_id else 0)

        if len(TallyList) > 0:
            OverallTally.append(1 if sum(TallyList) / len(TallyList) > 0.8 else 0)
            PerFrameTally.extend(TallyList)

    return PerFrameTally, OverallTally


def compute_metrics(inference_files, MetaDF, GTDataDir, 
                   ModelName="Model", IDType="IDAlgo", TrackingAlgo="Tracker"):
    """
    Optimized compute metrics for much better performance
    """    
    ValidationVideos = list(inference_files.keys())
    SaveName = f"{IDType}_{TrackingAlgo}_{ModelName}"
    
    # Collect all data using optimized functions
    all_peck_data = []
    all_proptime_data = []
    all_behavior_data = []
    all_frame_accuracies = []
    all_track_accuracies = []
    
    for vid in ValidationVideos:
        print(f"Processing {vid}...")
        
        # Load inference data for this video
        BehavDict = load_data(inference_files[vid]['behav_dict'])
        TrackedBBox = load_data(inference_files[vid]['tracked_bbox']) 
        IDMatch = load_data(inference_files[vid]['id_match'])
        
        # Load ground truth data
        GTBBox = load_data(os.path.join(GTDataDir, vid + "_BBox.csv"))
        GTBehav = load_data(os.path.join(GTDataDir, vid + "_PeckEvents.csv"))
        
        VideoLength = MetaDF.loc[MetaDF["Video"]==vid]["VideoLength"].values[0]  
        VideoMinutes = VideoLength/25/60
        
        # Use optimized functions
        peck_data = compute_feed_rates_optimized(GTBehav, BehavDict, IDMatch, VideoMinutes)
        proptime_data = compute_proptime_optimized(GTBBox.copy(), TrackedBBox.copy(), IDMatch, VideoLength)
        behavior_data = compute_behavior_metrics_optimized(GTBehav, BehavDict, IDMatch, VideoLength)
        frame_accuracy, track_accuracy = compute_frame_accuracy_optimized(GTBBox.copy(), TrackedBBox.copy(), IDMatch)

        # import ipdb; ipdb.set_trace()


        # Add metadata to results
        for item in peck_data:
            item.update({"Video": vid, "Model": ModelName, "IDAlgo": IDType, 
                        "Tracker": TrackingAlgo, "Name": SaveName})
        
        for item in proptime_data:
            item.update({"Video": vid, "Model": ModelName, "IDAlgo": IDType, 
                        "Tracker": TrackingAlgo, "Name": SaveName})
        
        for item in behavior_data:
            item.update({"Video": vid, "Model": ModelName, "IDAlgo": IDType, 
                        "Tracker": TrackingAlgo, "Name": SaveName})
        
        # Extend results
        all_peck_data.extend(peck_data)
        all_proptime_data.extend(proptime_data)
        all_behavior_data.extend(behavior_data)
        all_frame_accuracies.extend(frame_accuracy)
        all_track_accuracies.extend(track_accuracy)
    
    # Convert to output format
    PeckRateOut = {i: data for i, data in enumerate(all_peck_data)}
    PropTimeOut = {i: data for i, data in enumerate(all_proptime_data)}
    BehavPrecisionRecall = {i: data for i, data in enumerate(all_behavior_data)}
    
    # Compute overall accuracies (averaged across videos)
    frame_correct = np.mean(all_frame_accuracies) if all_frame_accuracies else 0
    track_correct = np.mean(all_track_accuracies) if all_track_accuracies else 0

    PerFramePercentageCorrect = {SaveName: frame_correct}
    TrackPercentageCorrect = {SaveName: track_correct}

    return {
        "PeckRateOut": PeckRateOut,
        "PropTimeOut": PropTimeOut,
        "BehavPrecisionRecall": BehavPrecisionRecall,
        "PerFramePercentageCorrect": PerFramePercentageCorrect,
        "TrackPercentageCorrect": TrackPercentageCorrect
    }


def main():
    parser = argparse.ArgumentParser(description='Compute metrics for all algorithm combinations in inference directory')
    parser.add_argument('--dataset_path', required=True, help='Path to dataset root directory')
    parser.add_argument('--inference_dir', required=True, help='Directory containing inference result files')
    parser.add_argument('--output_dir', default='./results', help='Output directory for results')
    parser.add_argument('--force', action='store_true', default=False, help='Force recomputation even if results exist')
    
    args = parser.parse_args()
    
    # Construct paths from dataset_path
    metadata_csv = os.path.join(args.dataset_path, 'ApplicationSpecific', 'MetaData.csv')
    gt_data_dir = os.path.join(args.dataset_path, 'ApplicationSpecific', 'GroundTruth')
    human_dir = os.path.join(args.dataset_path, 'ApplicationSpecific', 'HumanBenchmarkBORIS')
    
    # Verify paths exist
    if not os.path.exists(metadata_csv):
        print(f"Error: Metadata file not found at {metadata_csv}")
        return
    
    if not os.path.exists(gt_data_dir):
        print(f"Error: Ground truth directory not found at {gt_data_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover all algorithm combinations
    print(f"Discovering algorithm combinations in {args.inference_dir}...")
    combinations = discover_algorithm_combinations(args.inference_dir)
    
    if not combinations:
        print("No algorithm combinations found. Please check the inference directory.")
        return
        
    # Load metadata once
    print(f"Loading metadata from {metadata_csv}...")
    MetaDF = pd.read_csv(metadata_csv)
    if "PossibleBirds" in MetaDF.columns:
        MetaDF["PossibleBirds"] = MetaDF["PossibleBirds"].apply(literal_eval)
    
    # Process each combination
    processed_count = 0
    skipped_count = 0
    
    for i, (model_name, id_algo, tracker) in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] Processing combination: {model_name} + {id_algo} + {tracker}")
        
        # Check if results already exist
        if not args.force and check_results_exist(args.output_dir, model_name, id_algo, tracker):
            print(f"Results already exist for {id_algo}_{tracker}_{model_name}, skipping...")
            skipped_count += 1
            continue
        
        # Find inference files for this combination
        inference_files = find_inference_files(args.inference_dir, model_name, id_algo, tracker)
        
        if not inference_files:
            print(f"No complete sets of inference files found for this combination, skipping...")
            skipped_count += 1
            continue
            
        print(f"Found inference files for {len(inference_files)} videos: {list(inference_files.keys())}")

        # Warn if not all 12 videos were found
        if len(inference_files) != 12:
            print(f"⚠️  WARNING: Expected 12 videos but found {len(inference_files)} videos for {id_algo}_{tracker}_{model_name}")
            missing_videos = set(ValidationVideos) - set(inference_files.keys())
            if missing_videos:
                print(f"    Missing videos: {sorted(missing_videos)}")

        # Compute metrics
        print("Computing metrics...")
        try:
            results = compute_metrics(
                inference_files=inference_files,
                MetaDF=MetaDF,
                GTDataDir=gt_data_dir,
                ModelName=model_name,
                IDType=id_algo,
                TrackingAlgo=tracker
            )
            
            # Save results
            output_prefix = f"{id_algo}_{tracker}_{model_name}"
            
            with open(os.path.join(args.output_dir, f"{output_prefix}_metrics.pkl"), "wb") as f:
                pickle.dump(results, f)
            
            # Save individual metric DataFrames as CSV
            pd.DataFrame.from_dict(results["PeckRateOut"], orient='index').to_csv(
                os.path.join(args.output_dir, f"{output_prefix}_peck_rates.csv"), index=False)
            
            pd.DataFrame.from_dict(results["PropTimeOut"], orient='index').to_csv(
                os.path.join(args.output_dir, f"{output_prefix}_proptime.csv"), index=False)
            
            pd.DataFrame.from_dict(results["BehavPrecisionRecall"], orient='index').to_csv(
                os.path.join(args.output_dir, f"{output_prefix}_precision_recall.csv"), index=False)
            
            # Save summary metrics
            summary_metrics = {
                "PerFramePercentageCorrect": results["PerFramePercentageCorrect"],
                "TrackPercentageCorrect": results["TrackPercentageCorrect"]
            }

            with open(os.path.join(args.output_dir, f"{output_prefix}_summary.pkl"), "wb") as f:
                pickle.dump(summary_metrics, f)

            print(f"✓ Results saved: {output_prefix}")
            print(f"  Per Frame Percentage Correct: {list(results['PerFramePercentageCorrect'].values())[0]:.3f}")
            print(f"  Track Percentage Correct: {list(results['TrackPercentageCorrect'].values())[0]:.3f}")
            print(f"  Processed {len(results['PeckRateOut'])} peck rate measurements")
            print(f"  Processed {len(results['PropTimeOut'])} proportion time measurements")
            print(f"  Processed {len(results['BehavPrecisionRecall'])} behavior precision/recall measurements")
            
            processed_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {model_name} + {id_algo} + {tracker}: {str(e)}")
            skipped_count += 1
            continue
    
    print(f"\n=== Summary ===")
    print(f"Total combinations found: {len(combinations)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Results saved to: {args.output_dir}")
    
    # Compute human benchmark data if available
    human_benchmark_data = None
    if os.path.exists(human_dir):
        print(f"\n🧑 Computing human benchmark data from {human_dir}...")
        try:
            human_benchmark_data = GetHumanBench(human_dir, gt_data_dir, ValidationVideos)
            print("✅ Human benchmark data computed successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not compute human benchmark data: {str(e)}")
            human_benchmark_data = None
    else:
        print(f"⚠️  Human benchmark directory not found: {human_dir}")
    
    # Compute random benchmark data if available
    random_benchmark_data = None
    if os.path.exists(human_dir):
        print(f"\n🎲 Loading random benchmark data from {human_dir}...")
        try:
            random_benchmark_data = GetRandomBench(human_dir)
            if random_benchmark_data:
                print("✅ Random benchmark data loaded successfully")
            else:
                print("⚠️  Random benchmark CSV files not found")
        except Exception as e:
            print(f"⚠️  Warning: Could not load random benchmark data: {str(e)}")
            random_benchmark_data = None
    
    # Create comprehensive summary CSV from all results
    create_summary_csv(args.output_dir, human_benchmark_data, random_benchmark_data)


if __name__ == "__main__":
    main()