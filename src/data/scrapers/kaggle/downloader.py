"""
Kaggle Dataset Downloader and Processor

Automatically downloads and processes medical datasets from Kaggle
for training the medical AI model.
"""

import os
import zipfile
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

from ...utils.logging.logger import get_logger
from ...utils.config.loader import load_config

logger = get_logger(__name__)


class KaggleDatasetDownloader:
    """
    Downloads and processes medical datasets from Kaggle.
    
    Features:
    - Automatic dataset discovery
    - Batch downloading
    - Data validation and cleaning
    - Format standardization
    - Medical data preprocessing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path or "conf/kaggle_catalog.yaml")
        self.api = KaggleApi()
        self.api.authenticate()
        
        # Setup paths
        self.download_dir = Path("data/raw/kaggle")
        self.processed_dir = Path("data/processed/kaggle")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Kaggle Dataset Downloader initialized")
    
    def download_medical_datasets(self, dataset_list: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Download specified medical datasets from Kaggle.
        
        Args:
            dataset_list: List of dataset identifiers to download
            
        Returns:
            Dictionary mapping dataset names to download paths
        """
        if dataset_list is None:
            dataset_list = self.config.get("medical_datasets", [])
        
        downloaded_datasets = {}
        
        for dataset_id in dataset_list:
            try:
                logger.info(f"Downloading dataset: {dataset_id}")
                download_path = self._download_dataset(dataset_id)
                
                if download_path:
                    downloaded_datasets[dataset_id] = download_path
                    logger.info(f"Successfully downloaded: {dataset_id}")
                else:
                    logger.error(f"Failed to download: {dataset_id}")
                    
            except Exception as e:
                logger.error(f"Error downloading {dataset_id}: {e}")
                continue
        
        return downloaded_datasets
    
    def _download_dataset(self, dataset_id: str) -> Optional[str]:
        """Download a single dataset."""
        dataset_dir = self.download_dir / dataset_id.replace("/", "_")
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            # Download dataset
            self.api.dataset_download_files(
                dataset_id,
                path=str(dataset_dir),
                unzip=True
            )
            
            return str(dataset_dir)
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_id}: {e}")
            return None
    
    def process_medical_datasets(self, downloaded_datasets: Dict[str, str]) -> Dict[str, Any]:
        """
        Process downloaded medical datasets into standardized format.
        
        Args:
            downloaded_datasets: Dictionary of dataset names to paths
            
        Returns:
            Processing results and statistics
        """
        processing_results = {}
        
        for dataset_id, dataset_path in downloaded_datasets.items():
            try:
                logger.info(f"Processing dataset: {dataset_id}")
                result = self._process_single_dataset(dataset_id, dataset_path)
                processing_results[dataset_id] = result
                
            except Exception as e:
                logger.error(f"Error processing {dataset_id}: {e}")
                processing_results[dataset_id] = {"status": "error", "error": str(e)}
        
        return processing_results
    
    def _process_single_dataset(self, dataset_id: str, dataset_path: str) -> Dict[str, Any]:
        """Process a single dataset."""
        dataset_path = Path(dataset_path)
        
        # Identify dataset type and structure
        dataset_info = self._analyze_dataset_structure(dataset_path)
        
        # Apply dataset-specific processing
        if "medical-text" in dataset_id.lower():
            return self._process_medical_text_dataset(dataset_path, dataset_info)
        elif "medical-image" in dataset_id.lower() or "xray" in dataset_id.lower():
            return self._process_medical_image_dataset(dataset_path, dataset_info)
        elif "clinical" in dataset_id.lower():
            return self._process_clinical_dataset(dataset_path, dataset_info)
        else:
            return self._process_generic_medical_dataset(dataset_path, dataset_info)
    
    def _analyze_dataset_structure(self, dataset_path: Path) -> Dict[str, Any]:
        """Analyze dataset structure and content."""
        info = {
            "files": [],
            "total_size": 0,
            "file_types": {},
            "data_types": []
        }
        
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                info["files"].append(str(file_path.relative_to(dataset_path)))
                info["total_size"] += file_path.stat().st_size
                
                # Count file types
                suffix = file_path.suffix.lower()
                info["file_types"][suffix] = info["file_types"].get(suffix, 0) + 1
        
        # Determine data types
        if any(ext in info["file_types"] for ext in [".jpg", ".png", ".dcm", ".nii"]):
            info["data_types"].append("images")
        if any(ext in info["file_types"] for ext in [".csv", ".json", ".txt"]):
            info["data_types"].append("text")
        if ".xml" in info["file_types"]:
            info["data_types"].append("structured")
        
        return info
    
    def _process_medical_text_dataset(self, dataset_path: Path, info: Dict) -> Dict[str, Any]:
        """Process medical text datasets."""
        processed_data = []
        
        # Process CSV files
        for csv_file in dataset_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                
                # Standardize column names
                df.columns = df.columns.str.lower().str.replace(" ", "_")
                
                # Extract medical text content
                text_columns = [col for col in df.columns if any(
                    keyword in col for keyword in ["text", "description", "note", "report", "summary"]
                )]
                
                for _, row in df.iterrows():
                    record = {
                        "source": str(csv_file.name),
                        "type": "medical_text"
                    }
                    
                    # Extract text content
                    for col in text_columns:
                        if pd.notna(row[col]):
                            record[col] = str(row[col])
                    
                    # Extract metadata
                    for col in df.columns:
                        if col not in text_columns and pd.notna(row[col]):
                            record[f"meta_{col}"] = row[col]
                    
                    processed_data.append(record)
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
        
        # Save processed data
        output_file = self.processed_dir / f"{dataset_path.name}_processed.json"
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
        
        return {
            "status": "success",
            "records_processed": len(processed_data),
            "output_file": str(output_file)
        }
    
    def _process_medical_image_dataset(self, dataset_path: Path, info: Dict) -> Dict[str, Any]:
        """Process medical image datasets."""
        image_data = []
        
        # Find image files and associated metadata
        image_extensions = [".jpg", ".jpeg", ".png", ".dcm", ".nii", ".tiff"]
        
        for img_file in dataset_path.rglob("*"):
            if img_file.suffix.lower() in image_extensions:
                record = {
                    "image_path": str(img_file.relative_to(dataset_path)),
                    "filename": img_file.name,
                    "size": img_file.stat().st_size,
                    "type": "medical_image",
                    "format": img_file.suffix.lower()
                }
                
                # Look for associated metadata files
                metadata_file = img_file.with_suffix(".json")
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        record["metadata"] = metadata
                    except:
                        pass
                
                image_data.append(record)
        
        # Save processed data
        output_file = self.processed_dir / f"{dataset_path.name}_images.json"
        with open(output_file, 'w') as f:
            json.dump(image_data, f, indent=2, default=str)
        
        return {
            "status": "success",
            "images_processed": len(image_data),
            "output_file": str(output_file)
        }
    
    def _process_clinical_dataset(self, dataset_path: Path, info: Dict) -> Dict[str, Any]:
        """Process clinical datasets with structured medical data."""
        clinical_data = []
        
        for csv_file in dataset_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                
                # Standardize medical data
                for _, row in df.iterrows():
                    record = {
                        "source": str(csv_file.name),
                        "type": "clinical_data"
                    }
                    
                    # Process each column
                    for col, value in row.items():
                        if pd.notna(value):
                            # Standardize medical terminology
                            if any(keyword in col.lower() for keyword in 
                                  ["diagnosis", "symptom", "condition", "disease"]):
                                record[f"medical_{col.lower()}"] = str(value)
                            elif any(keyword in col.lower() for keyword in 
                                    ["age", "gender", "sex", "weight", "height"]):
                                record[f"demographic_{col.lower()}"] = value
                            else:
                                record[col.lower()] = value
                    
                    clinical_data.append(record)
                
            except Exception as e:
                logger.error(f"Error processing clinical file {csv_file}: {e}")
        
        # Save processed data
        output_file = self.processed_dir / f"{dataset_path.name}_clinical.json"
        with open(output_file, 'w') as f:
            json.dump(clinical_data, f, indent=2, default=str)
        
        return {
            "status": "success",
            "records_processed": len(clinical_data),
            "output_file": str(output_file)
        }
    
    def _process_generic_medical_dataset(self, dataset_path: Path, info: Dict) -> Dict[str, Any]:
        """Process generic medical datasets."""
        # Generic processing for unknown dataset types
        return {
            "status": "processed_generic",
            "files_found": len(info["files"]),
            "file_types": info["file_types"],
            "data_types": info["data_types"]
        }
    
    def get_recommended_datasets(self) -> List[Dict[str, str]]:
        """Get list of recommended medical datasets from Kaggle."""
        recommended = [
            # Medical Text & QA Datasets
            {
                "id": "evidence/medqa-usmle",
                "name": "MedQA USMLE",
                "type": "medical_qa",
                "description": "USMLE medical question answering dataset"
            },
            {
                "id": "jithinanievarghese/drugs-related-to-common-treatments",
                "name": "Drugs Related to Common Treatments",
                "type": "drug_data",
                "description": "Drug information and treatment relationships"
            },
            {
                "id": "matthewjansen/pubmed-200k-rtc",
                "name": "PubMed 200K RTC",
                "type": "medical_text",
                "description": "200K PubMed randomized controlled trials"
            },
            {
                "id": "singhnavjot2062001/11000-medicine-details",
                "name": "11000 Medicine Details",
                "type": "drug_data",
                "description": "Comprehensive medicine database"
            },
            {
                "id": "ujjwalaggarwal402/medicine-dataset",
                "name": "Medicine Dataset",
                "type": "drug_data",
                "description": "Medicine information and classifications"
            },
            {
                "id": "shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes",
                "name": "250K Medicines Usage & Side Effects",
                "type": "drug_data",
                "description": "Comprehensive drug usage, side effects, and substitutes"
            },
            {
                "id": "cahyaasrini/openfda-human-otc-drug-labels",
                "name": "OpenFDA Human OTC Drug Labels",
                "type": "otc_drugs",
                "description": "Over-the-counter drug labels and information"
            },
            
            # Medical Imaging Datasets
            {
                "id": "kmader/siim-medical-images",
                "name": "SIIM Medical Images",
                "type": "medical_imaging",
                "description": "Society for Imaging Informatics in Medicine dataset"
            },
            {
                "id": "andrewmvd/covid19-ct-scans",
                "name": "COVID-19 CT Scans",
                "type": "ct_imaging",
                "description": "CT scans for COVID-19 detection"
            },
            {
                "id": "hgunraj/covidxct",
                "name": "COVIDx-CT",
                "type": "ct_imaging",
                "description": "Large-scale CT scan dataset for COVID-19"
            },
            {
                "id": "mohamedhanyyy/chest-ctscan-images",
                "name": "Chest CT Scan Images",
                "type": "ct_imaging",
                "description": "Chest CT scans for various conditions"
            },
            {
                "id": "felipekitamura/head-ct-hemorrhage",
                "name": "Head CT Hemorrhage",
                "type": "ct_imaging",
                "description": "Head CT scans for hemorrhage detection"
            },
            {
                "id": "plameneduardo/sarscov2-ctscan-dataset",
                "name": "SARS-CoV-2 CT Scan Dataset",
                "type": "ct_imaging",
                "description": "CT scans for SARS-CoV-2 detection"
            },
            {
                "id": "ozguraslank/brain-stroke-ct-dataset",
                "name": "Brain Stroke CT Dataset",
                "type": "ct_imaging",
                "description": "CT scans for brain stroke detection"
            },
            {
                "id": "abbymorgan/cranial-ct",
                "name": "Cranial CT",
                "type": "ct_imaging",
                "description": "Cranial CT scan dataset"
            },
            {
                "id": "murtozalikhon/brain-tumor-multimodal-image-ct-and-mri",
                "name": "Brain Tumor Multimodal (CT & MRI)",
                "type": "multimodal_imaging",
                "description": "Brain tumor detection with CT and MRI"
            },
            {
                "id": "andrewmvd/liver-tumor-segmentation",
                "name": "Liver Tumor Segmentation",
                "type": "ct_imaging",
                "description": "Liver tumor segmentation dataset"
            },
            {
                "id": "mateuszbuda/lgg-mri-segmentation",
                "name": "LGG MRI Segmentation",
                "type": "mri_imaging",
                "description": "Lower Grade Glioma MRI segmentation"
            },
            
            # Blood Test & Lab Data
            {
                "id": "ehababoelnaga/anemia-types-classification",
                "name": "Anemia Types Classification",
                "type": "lab_data",
                "description": "Blood test data for anemia classification"
            },
            {
                "id": "ahmedelsayedtaha/complete-blood-count-cbc-test",
                "name": "Complete Blood Count (CBC) Test",
                "type": "lab_data",
                "description": "CBC test results and analysis"
            },
            {
                "id": "orvile/complete-blood-count-cbc-dataset",
                "name": "Complete Blood Count Dataset",
                "type": "lab_data",
                "description": "Comprehensive CBC dataset"
            },
            {
                "id": "mdnoukhej/complete-blood-count-cbc",
                "name": "CBC Medical Dataset",
                "type": "lab_data",
                "description": "Medical CBC test dataset"
            }
        ]
        
        return recommended
    
    def download_and_process_all(self) -> Dict[str, Any]:
        """Download and process all recommended medical datasets."""
        logger.info("Starting batch download and processing of medical datasets")
        
        # Get recommended datasets
        recommended = self.get_recommended_datasets()
        dataset_ids = [dataset["id"] for dataset in recommended]
        
        # Download datasets
        downloaded = self.download_medical_datasets(dataset_ids)
        
        # Process datasets
        processed = self.process_medical_datasets(downloaded)
        
        # Generate summary
        summary = {
            "total_datasets": len(dataset_ids),
            "successfully_downloaded": len(downloaded),
            "successfully_processed": len([r for r in processed.values() if r.get("status") == "success"]),
            "download_results": downloaded,
            "processing_results": processed
        }
        
        logger.info(f"Batch processing complete: {summary}")
        return summary


def main():
    """Main function for Kaggle dataset processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process Kaggle medical datasets")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to download")
    parser.add_argument("--all", action="store_true", help="Download all recommended datasets")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    downloader = KaggleDatasetDownloader(config_path=args.config)
    
    if args.all:
        results = downloader.download_and_process_all()
    elif args.datasets:
        downloaded = downloader.download_medical_datasets(args.datasets)
        results = downloader.process_medical_datasets(downloaded)
    else:
        # Show recommended datasets
        recommended = downloader.get_recommended_datasets()
        print("Recommended medical datasets:")
        for dataset in recommended:
            print(f"- {dataset['id']}: {dataset['name']} ({dataset['type']})")
        return
    
    print(f"Processing complete: {results}")


if __name__ == "__main__":
    main()
