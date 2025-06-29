import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, Dict, List
import json
from pathlib import Path
from datetime import datetime
import hashlib
from loguru import logger


class S3ModelManager:
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            
            self._ensure_bucket_exists()
            logger.info(f"S3 client initialized for bucket: {bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise
    
    def _ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    logger.info(f"Created bucket: {self.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    raise
            else:
                logger.error(f"Error checking bucket: {e}")
                raise
    
    def upload_model(self, client_id: str, model_path: str, metadata: Dict = None) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = Path(model_path).name
            s3_key = f"models/{client_id}/{timestamp}_{model_filename}"
            
            upload_metadata = {
                'client_id': client_id,
                'upload_timestamp': datetime.now().isoformat(),
                'original_filename': model_filename,
                'file_size': str(os.path.getsize(model_path))
            }
            
            if metadata:
                upload_metadata.update(metadata)
            
            with open(model_path, 'rb') as file:
                self.s3_client.upload_fileobj(
                    file,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={
                        'Metadata': {k: str(v) for k, v in upload_metadata.items()},
                        'ContentType': 'application/octet-stream'
                    }
                )
            
            metadata_key = f"models/{client_id}/{timestamp}_metadata.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(upload_metadata, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Model uploaded successfully: s3://{self.bucket_name}/{s3_key}")
            return s3_key
            
        except Exception as e:
            logger.error(f"Failed to upload model for {client_id}: {str(e)}")
            raise
    
    def download_model(self, client_id: str, s3_key: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Model downloaded: {s3_key} -> {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model {s3_key}: {str(e)}")
            return False
    
    def list_client_models(self, client_id: str) -> List[Dict]:
        try:
            prefix = f"models/{client_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            models = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.pt') or obj['Key'].endswith('.pth'):
                        metadata_response = self.s3_client.head_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
                        
                        model_info = {
                            's3_key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'metadata': metadata_response.get('Metadata', {})
                        }
                        models.append(model_info)
            
            logger.info(f"Found {len(models)} models for client {client_id}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models for {client_id}: {str(e)}")
            return []
    
    def delete_model(self, s3_key: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            
            metadata_key = s3_key.replace('.pt', '_metadata.json').replace('.pth', '_metadata.json')
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=metadata_key)
            except ClientError:
                pass
            
            logger.info(f"Model deleted: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {s3_key}: {str(e)}")
            return False
    
    def get_model_metadata(self, s3_key: str) -> Optional[Dict]:
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response.get('Metadata', {})
        except Exception as e:
            logger.error(f"Failed to get metadata for {s3_key}: {str(e)}")
            return None
    
    def create_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {s3_key}: {str(e)}")
            return None


class S3ModelTrainer:
    
    def __init__(self, config, s3_manager: S3ModelManager):
        from ..services.trainer import ModelTrainer
        self.trainer = ModelTrainer(config)
        self.s3_manager = s3_manager
        self.config = config
    
    def train_and_upload(self, raw_data: List[Dict]) -> Dict:
        try:
            results = self.trainer.train(raw_data)
            
            metadata = {
                'architecture': self.config.architecture,
                'sentiment_levels': self.config.sentiment_levels,
                'vocab_size': results['vocab_size'],
                'training_time': results['training_time'],
                'best_accuracy': results['metrics']['best_val_accuracy'],
                'total_epochs': results['total_epochs']
            }
            
            local_model_path = results['model_path']
            s3_key = self.s3_manager.upload_model(
                client_id=self.config.client_id,
                model_path=local_model_path,
                metadata=metadata
            )
            
            results['s3_key'] = s3_key
            results['s3_bucket'] = self.s3_manager.bucket_name
            results['model_url'] = self.s3_manager.create_presigned_url(s3_key)
            
            logger.info(f"Model trained and uploaded to S3: {s3_key}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to train and upload model: {str(e)}")
            raise
    
    def load_from_s3(self, s3_key: str):
        try:
            local_path = f"/tmp/{os.path.basename(s3_key)}"
            
            if self.s3_manager.download_model(self.config.client_id, s3_key, local_path):
                model = self.trainer.load_model(local_path)
                
                os.remove(local_path)
                
                return model
            else:
                raise Exception("Failed to download model from S3")
                
        except Exception as e:
            logger.error(f"Failed to load model from S3: {str(e)}")
            raise


def get_s3_config() -> Dict[str, str]:
    return {
        'bucket_name': os.getenv('S3_BUCKET_NAME', 'sentiment-ai-models'),
        'region': os.getenv('AWS_REGION', 'us-east-1'),
        'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY')
    }