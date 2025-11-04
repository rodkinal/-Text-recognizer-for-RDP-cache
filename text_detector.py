#!/usr/bin/env python3
"""
Text Detection and Classification Script

This script scans images from the 'images-2-tiles/processed_images' directory,
detects text using EasyOCR (open source OCR library), and copies images 
containing text to the 'tiles-with-text' directory.

Features:
- Uses EasyOCR for text detection (alternative to Pytesseract)
- Supports multiple languages (English and Spanish by default)
- Configurable confidence threshold for text detection
- Detailed logging of processing results
- Progress tracking with visual indicators
- Generates comprehensive reports

Author: Rodkinal
Date: November 2, 2025
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import argparse

try:
    import easyocr
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required packages using: pip install -r requirements.txt")
    sys.exit(1)


class TextDetector:
    """Main class for text detection and image classification"""
    
    def __init__(self, 
                 source_dir: str, 
                 output_dir: str,
                 confidence_threshold: float = 0.5,
                 languages: List[str] = ['en', 'es']):
        """
        Initialize the TextDetector
        
        Args:
            source_dir: Path to source images directory
            output_dir: Path to output directory for images with text
            confidence_threshold: Minimum confidence score for text detection
            languages: List of language codes for OCR
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.languages = languages
        
        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'images_with_text': 0,
            'images_without_text': 0,
            'errors': 0,
            'processing_time': 0
        }
        
        # Setup logging
        self.setup_logging()
        
        # Initialize EasyOCR reader
        print(f"DEBUG: Initializing EasyOCR with languages: {languages}")
        self.logger.info(f"Initializing EasyOCR with languages: {languages}")
        try:
            print("DEBUG: Creating EasyOCR Reader...")
            self.reader = easyocr.Reader(languages, gpu=False)  # Set gpu=True if CUDA available
            print("DEBUG: EasyOCR Reader created successfully")
            self.logger.info("EasyOCR initialized successfully")
        except Exception as e:
            print(f"DEBUG: Failed to initialize EasyOCR: {e}")
            self.logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
        
        # Create output directories
        self.create_directories()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.output_dir.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'text_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_directories(self):
        """Create necessary output directories"""
        directories = [
            self.output_dir,
            self.output_dir.parent / 'logs',
            self.output_dir.parent / 'reports'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created/verified directory: {directory}")
    
    def get_image_files(self, limit: Optional[int] = None) -> List[Path]:
        """Get list of image files from source directory"""
        print(f"DEBUG: Checking source directory: {self.source_dir}")
        print(f"DEBUG: Directory exists: {self.source_dir.exists()}")
        
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        
        # Supported image extensions
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        image_files = []
        for ext in extensions:
            files_found = list(self.source_dir.glob(f'*{ext}'))
            files_found_upper = list(self.source_dir.glob(f'*{ext.upper()}'))
            print(f"DEBUG: Found {len(files_found)} files with extension {ext}")
            print(f"DEBUG: Found {len(files_found_upper)} files with extension {ext.upper()}")
            image_files.extend(files_found)
            image_files.extend(files_found_upper)
        
        # Filter out metadata files
        image_files = [f for f in image_files if not f.name.endswith('_metadata.json')]
        
        # Apply limit for testing
        if limit and len(image_files) > limit:
            image_files = image_files[:limit]
            print(f"DEBUG: Limited to {limit} images for testing")
        
        print(f"DEBUG: Total image files to process: {len(image_files)}")
        if image_files:
            print(f"DEBUG: First few files: {[f.name for f in image_files[:3]]}")
        
        self.logger.info(f"Found {len(image_files)} image files in {self.source_dir}")
        return sorted(image_files)
    
    def preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Preprocess image for better OCR results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array or None if error
        """
        try:
            # Read image with OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Could not read image: {image_path}")
                return None
            
            # Convert to RGB (EasyOCR expects RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Optional: Apply some preprocessing for better OCR
            # Resize if image is too small (helps with small text)
            height, width = image.shape[:2]
            if height < 100 or width < 100:
                scale_factor = max(100 / height, 100 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def detect_text(self, image: np.ndarray) -> Tuple[bool, List[Dict], str]:
        """
        Detect text in image using EasyOCR
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            Tuple of (has_text, detection_results, extracted_text)
        """
        try:
            # Perform OCR detection
            results = self.reader.readtext(image)
            
            # Process results
            detected_texts = []
            has_significant_text = False
            all_text = []
            
            for (bbox, text, confidence) in results:
                if confidence >= self.confidence_threshold:
                    # Filter out very short strings or noise
                    if len(text.strip()) >= 2 and text.strip().isalnum() or any(c.isalpha() for c in text):
                        detected_texts.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        all_text.append(text.strip())
                        has_significant_text = True
            
            combined_text = ' '.join(all_text) if all_text else ''
            
            return has_significant_text, detected_texts, combined_text
            
        except Exception as e:
            self.logger.error(f"Error during text detection: {e}")
            return False, [], ''
    
    def copy_image_with_metadata(self, source_path: Path, text_info: Dict):
        """Copy image to output directory without individual metadata files"""
        try:
            # Copy image file only
            dest_path = self.output_dir / source_path.name
            shutil.copy2(source_path, dest_path)
            
            self.logger.info(f"Copied image with text: {source_path.name}")
            
            # Store metadata for the consolidated report (optional)
            if not hasattr(self, '_image_metadata'):
                self._image_metadata = []
            
            # Convert numpy types to JSON serializable types for the report
            json_safe_texts = []
            for item in text_info['detected_texts']:
                json_safe_item = {
                    'text': item['text'],
                    'confidence': float(item['confidence']),
                    'bbox': [[float(x), float(y)] for x, y in item['bbox']]
                }
                json_safe_texts.append(json_safe_item)
            
            image_metadata = {
                'source_path': str(source_path),
                'filename': source_path.name,
                'detection_timestamp': datetime.now().isoformat(),
                'text_detected': json_safe_texts,
                'combined_text': text_info['combined_text']
            }
            
            self._image_metadata.append(image_metadata)
            
        except Exception as e:
            self.logger.error(f"Error copying image {source_path}: {e}")
            self.stats['errors'] += 1
    
    def process_single_image(self, image_path: Path) -> bool:
        """
        Process a single image for text detection
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if text was detected, False otherwise
        """
        try:
            self.logger.debug(f"Processing: {image_path.name}")
            
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                self.stats['errors'] += 1
                return False
            
            # Detect text
            has_text, detected_texts, combined_text = self.detect_text(image)
            
            if has_text:
                # Copy image with metadata
                text_info = {
                    'detected_texts': detected_texts,
                    'combined_text': combined_text
                }
                self.copy_image_with_metadata(image_path, text_info)
                self.stats['images_with_text'] += 1
                return True
            else:
                self.stats['images_without_text'] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def process_all_images(self, limit: Optional[int] = None):
        """Process all images in the source directory"""
        start_time = datetime.now()
        print("=" * 60)
        print("STARTING TEXT DETECTION PROCESS")
        print("=" * 60)
        self.logger.info("Starting text detection process...")
        
        image_files = self.get_image_files(limit)
        self.stats['total_images'] = len(image_files)
        
        if not image_files:
            print("WARNING: No image files found to process")
            self.logger.warning("No image files found to process")
            return
        
        # Process images with progress tracking
        for i, image_path in enumerate(image_files, 1):
            print(f"Processing image {i}/{len(image_files)}: {image_path.name}")
            self.process_single_image(image_path)
            
            # Progress indicator
            if i % 10 == 0 or i == len(image_files):
                progress = (i / len(image_files)) * 100
                print(f"Progress: {progress:.1f}% ({i}/{len(image_files)})")
        
        # Calculate processing time
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        self.generate_report()
        self.print_summary()
    
    def generate_report(self):
        """Generate detailed processing report"""
        report_dir = self.output_dir.parent / 'reports'
        report_file = report_dir / f'text_detection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        report = {
            'execution_info': {
                'timestamp': datetime.now().isoformat(),
                'source_directory': str(self.source_dir),
                'output_directory': str(self.output_dir),
                'confidence_threshold': float(self.confidence_threshold),
                'languages': self.languages
            },
            'statistics': self.stats,
            'performance': {
                'avg_time_per_image': self.stats['processing_time'] / max(self.stats['total_images'], 1),
                'images_per_second': self.stats['total_images'] / max(self.stats['processing_time'], 1)
            },
            'images_with_text': getattr(self, '_image_metadata', [])
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Report generated: {report_file}")
        
        # Also generate a simple list of images with text
        if hasattr(self, '_image_metadata') and self._image_metadata:
            images_list_file = self.output_dir / 'images_with_text.txt'
            with open(images_list_file, 'w', encoding='utf-8') as f:
                f.write(f"Images with detected text (confidence >= {self.confidence_threshold}):\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")
                
                for img_data in self._image_metadata:
                    f.write(f"File: {img_data['filename']}\n")
                    f.write(f"Text: {img_data['combined_text']}\n")
                    f.write(f"Regions: {len(img_data['text_detected'])}\n")
                    for region in img_data['text_detected']:
                        f.write(f"  - '{region['text']}' (confidence: {region['confidence']:.2f})\n")
                    f.write("\n")
            
            self.logger.info(f"Images list generated: {images_list_file}")
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*60)
        print("TEXT DETECTION PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"Images with text detected: {self.stats['images_with_text']}")
        print(f"Images without text: {self.stats['images_without_text']}")
        print(f"Processing errors: {self.stats['errors']}")
        print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        if self.stats['total_images'] > 0:
            print(f"Success rate: {((self.stats['total_images'] - self.stats['errors']) / self.stats['total_images']) * 100:.1f}%")
            print(f"Text detection rate: {(self.stats['images_with_text'] / self.stats['total_images']) * 100:.1f}%")
        print(f"Output directory: {self.output_dir}")
        print("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Detect text in images and classify them')
    
    # Input/Output directories
    parser.add_argument('--input', type=str, 
                       default='../images-2-tiles/processed_images',
                       help='Input directory containing images to process')
    
    parser.add_argument('--output', type=str,
                       default='tiles-with-text',
                       help='Output directory for images with text')
    parser.add_argument('--output-dir', type=str, dest='output',
                       help='Output directory for images with text (alias for --output)')
    
    # Processing parameters
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for text detection (0.0-1.0, default: 0.5)')
    parser.add_argument('--languages', nargs='+', default=['en', 'es'],
                       help='Languages for OCR (default: en es)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of images to process (for testing)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with only 5 images')
    
    args = parser.parse_args()
    
    # Set limit for test mode
    if args.test:
        args.limit = 5
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent.absolute()
    
    # Resolve paths
    if not Path(args.input).is_absolute():
        source_dir = script_dir / args.input
    else:
        source_dir = Path(args.input)
    
    if not Path(args.output).is_absolute():
        output_dir = script_dir / args.output
    else:
        output_dir = Path(args.output)
    
    print("Text Detection and Classification System")
    print("="*50)
    print(f"Input directory:  {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Languages: {args.languages}")
    if args.limit:
        print(f"Processing limit: {args.limit} images")
    if args.test:
        print("Running in TEST mode")
    print("="*50)
    
    try:
        # Initialize and run text detector
        detector = TextDetector(
            source_dir=source_dir,
            output_dir=output_dir,
            confidence_threshold=args.confidence,
            languages=args.languages
        )
        
        detector.process_all_images(limit=args.limit)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
