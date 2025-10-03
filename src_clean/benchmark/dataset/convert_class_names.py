#!/usr/bin/env python3
"""
Convert class_str from numeric strings to descriptive class names.
This script updates HuggingFace datasets to use meaningful class names instead of numeric IDs.

Usage:
    python convert_class_names.py --input-path /path/to/dataset --output-path /path/to/converted_dataset
    python convert_class_names.py --input-path /path/to/dataset --in-place  # Modify in place
"""
import argparse
import os
import sys
from datasets import load_from_disk, DatasetDict
from functools import partial


def get_class_name_mapping():
    """
    Get the mapping from numeric class IDs to descriptive class names.
    
    Returns:
        dict: Mapping from numeric string to descriptive name
    """
    class_name_mapping = {
        "1": "EW",          # Eclipsing Binary (EW type - contact binary)
        "2": "EA",          # Eclipsing Binary (EA type - detached binary) 
        "3": "\u03B2 Lyrae",
        "4": "RRab",        # RR Lyrae (ab type - fundamental mode pulsator)
        "5": "RRc",         # RR Lyrae (c type - first overtone pulsator)
        "6": "RRd",         # RR Lyrae (d type - double mode pulsator)
        "7": "Blazhko",    # Blazhko variable star
        "8": "RS CVn",      # RS Canum Venaticorum (spotted star with rotational modulation)
        "9": "ACEP",        
        "10": "Cep-II",      
        "11": "HADS",
        "12": "LADS",       
        "13": "LPV",
        "14": "ELL",
        "15": "Hump",
        "16": "PCEB",
        "17": "EAᵤₚ",         
    }
    return class_name_mapping


def convert_class_str(example, mapping):
    """
    Convert a single example's class_str from numeric ID to descriptive name.
    
    Args:
        example: Dataset example dictionary
        mapping: Dictionary mapping numeric strings to descriptive names
        
    Returns:
        dict: Updated example with converted class_str
    """
    old_class = example["class_str"]
    if old_class in mapping:
        example["class_str"] = mapping[old_class]
        return example
    else:
        raise ValueError(f"Unknown class ID: {old_class}. Expected one of {list(mapping.keys())}")


def convert_dataset(dataset, mapping, num_proc=4):
    """
    Convert all splits in a dataset to use descriptive class names.
    
    Args:
        dataset: HuggingFace DatasetDict
        mapping: Dictionary mapping numeric strings to descriptive names
        num_proc: Number of processes for parallel conversion
        
    Returns:
        DatasetDict: Dataset with converted class names
    """
    print("Converting class names from numeric IDs to descriptive names...")
    print(f"Mapping: {mapping}")
    
    converted_dataset = DatasetDict()
    
    for split_name, split_dataset in dataset.items():
        print(f"\nProcessing split: {split_name}")
        print(f"  Original size: {len(split_dataset)}")
        
        # Check original class distribution
        if "class_str" in split_dataset.features:
            original_classes = set(split_dataset["class_str"])
            print(f"  Original classes: {sorted(original_classes)}")
            
            # Convert the split
            converted_split = split_dataset.map(
                partial(convert_class_str, mapping=mapping),
                num_proc=num_proc,
                desc=f"Converting {split_name} class names"
            )
            
            # Check new class distribution
            new_classes = set(converted_split["class_str"])
            print(f"  New classes: {sorted(new_classes)}")
            print(f"  Conversion successful: {len(converted_split)} samples")
            
            converted_dataset[split_name] = converted_split
        else:
            print(f"  Warning: No 'class_str' field found in {split_name}, copying as-is")
            converted_dataset[split_name] = split_dataset
    
    return converted_dataset


def validate_conversion(original_dataset, converted_dataset, mapping):
    """
    Validate conversion by comparing class distributions before and after.
    
    Args:
        original_dataset: Original DatasetDict
        converted_dataset: Converted DatasetDict
        mapping: Mapping dictionary used for conversion
    """
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION VALIDATION")
    print("="*50)
    
    for split_name in original_dataset.keys():
        if split_name not in converted_dataset:
            print(f"❌ Split {split_name} missing from converted dataset")
            continue
            
        orig_split = original_dataset[split_name]
        conv_split = converted_dataset[split_name]
        
        if len(orig_split) != len(conv_split):
            print(f"❌ {split_name}: Size mismatch ({len(orig_split)} vs {len(conv_split)})")
            continue
        
        # Count samples per class before conversion
        from collections import Counter
        orig_counts = Counter(orig_split["class_str"])
        conv_counts = Counter(conv_split["class_str"])
        
        print(f"\n📊 {split_name} ({len(orig_split)} samples):")
        print(f"   Class mapping and counts:")
        
        # Show side-by-side mapping with counts
        counts_match = True
        for orig_class in sorted(orig_counts.keys()):
            orig_count = orig_counts[orig_class]
            if orig_class in mapping:
                expected_class = mapping[orig_class]
                conv_count = conv_counts.get(expected_class, 0)
                
                # Show the mapping with counts
                status = "✅" if orig_count == conv_count else "❌"
                print(f"     '{orig_class}' ({orig_count:,}) → '{expected_class}' ({conv_count:,}) {status}")
                
                if orig_count != conv_count:
                    counts_match = False
            else:
                print(f"     '{orig_class}' ({orig_count:,}) → [NO MAPPING] ❌")
                counts_match = False
        
        if counts_match:
            print(f"   ✅ All class counts match after conversion")
        else:
            print(f"   ❌ Class count mismatches detected!")


def main():
    parser = argparse.ArgumentParser(description="Convert class_str from numeric IDs to descriptive names")
    parser.add_argument("--input-path", type=str, required=True,
                       help="Path to input HuggingFace dataset directory")
    parser.add_argument("--output-path", type=str, default=None,
                       help="Path to save converted dataset (if not specified, uses input-path + '_descriptive_classes')")
    parser.add_argument("--in-place", action="store_true",
                       help="Modify the dataset in place (overwrites original)")
    parser.add_argument("--num-proc", type=int, default=4,
                       help="Number of processes for parallel conversion")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be converted without making changes")
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"❌ Error: Input path does not exist: {args.input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.in_place:
        output_path = args.input_path
        print(f"⚠️  WARNING: Will modify dataset in place: {output_path}")
    else:
        if args.output_path:
            output_path = args.output_path
        else:
            output_path = args.input_path.rstrip('/') + '_descriptive_classes'
        print(f"📁 Will save converted dataset to: {output_path}")
    
    # Load dataset
    print(f"📥 Loading dataset from: {args.input_path}")
    try:
        dataset = load_from_disk(args.input_path)
        print(f"✅ Successfully loaded dataset with splits: {list(dataset.keys())}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Get mapping
    mapping = get_class_name_mapping()
    
    # Show current state
    print(f"\n📊 Current dataset info:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")
        if "class_str" in split_data.features:
            classes = sorted(set(split_data["class_str"]))
            print(f"    Classes: {classes}")
        else:
            print(f"    ⚠️  No 'class_str' field found")
    
    # Dry run check
    if args.dry_run:
        print(f"\n🔍 DRY RUN - Would apply these conversions:")
        for old, new in mapping.items():
            print(f"  '{old}' → '{new}'")
        print(f"\nNo changes made. Remove --dry-run to perform conversion.")
        return
    
    # Perform conversion
    print(f"\n🔄 Converting class names...")
    try:
        converted_dataset = convert_dataset(dataset, mapping, args.num_proc)
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        sys.exit(1)
    
    # Validate conversion
    validate_conversion(dataset, converted_dataset, mapping)
    
    # Save converted dataset
    if not args.in_place and os.path.exists(output_path):
        response = input(f"\n⚠️  Output path already exists: {output_path}\nOverwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Conversion cancelled.")
            sys.exit(1)
    
    print(f"\n💾 Saving converted dataset to: {output_path}")
    try:
        converted_dataset.save_to_disk(output_path)
        print(f"✅ Successfully saved converted dataset!")
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
        sys.exit(1)
    
    # Final summary
    print(f"\n🎉 CONVERSION COMPLETE!")
    print(f"📁 Input:  {args.input_path}")
    print(f"📁 Output: {output_path}")
    print(f"🔄 Converted class names:")
    for old, new in mapping.items():
        print(f"   '{old}' → '{new}'")


if __name__ == "__main__":
    main()
