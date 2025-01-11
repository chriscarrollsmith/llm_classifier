import pytest
import asyncio
import pandas as pd
from enum import Enum
from pydantic import BaseModel
from unittest.mock import patch, MagicMock
from classifier import parse_llm_json_response, classify_text, classify_all, process_csv, TemplateError


# Test enums
class PrimaryClassification(str, Enum):
    TYPE_A = "type_a"
    TYPE_B = "type_b"

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ProjectType(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"

# Test nested model
class ClassificationDetails(BaseModel):
    primary: PrimaryClassification
    confidence: ConfidenceLevel
    project_type: ProjectType

class NestedClassificationResponse(BaseModel):
    classification: ClassificationDetails
    justification: str
    evidence: str


class TestResponse(BaseModel):
    reason: str
    classification: str


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'item': ['test item'],
        'category': ['test category']
    })


@pytest.fixture
def test_files(tmp_path):
    input_file = tmp_path / "test_input.csv"
    output_file = tmp_path / "test_output.csv"
    return input_file, output_file


def test_parse_llm_json_response_direct_json():
    """Test parsing direct JSON response"""
    content = '{"reason": "test reason", "classification": "person"}'
    result = parse_llm_json_response(content, TestResponse)
    assert result.reason == "test reason"
    assert result.classification == "person"


def test_parse_llm_json_response_markdown_json():
    """Test parsing JSON within markdown code fence"""
    content = '```json\n{"reason": "test reason", "classification": "place"}\n```'
    result = parse_llm_json_response(content, TestResponse)
    assert result.reason == "test reason"
    assert result.classification == "place"


@pytest.mark.asyncio
async def test_classify_text():
    """Test the classify_text function"""
    mock_response = {
        'choices': [{
            'message': {
                'content': '{"reason": "test reason", "classification": "thing"}'
            }
        }]
    }
    
    with patch('classifier.acompletion', return_value=mock_response):
        result = await classify_text("test prompt", TestResponse)
        assert result.reason == "test reason"
        assert result.classification == "thing"


@pytest.mark.asyncio
async def test_classify_all(sample_df):
    """Test the classify_all function"""
    mock_response = TestResponse(reason="test reason", classification="person")
    
    with patch('classifier.classify_text', return_value=mock_response):
        prompt_template = "Classify {item} in {category}"
        results = await classify_all(sample_df, prompt_template, TestResponse)
        assert len(results) == 1
        assert results[0].reason == "test reason"
        assert results[0].classification == "person"


def test_classify_all_invalid_template():
    """Test classify_all with invalid template placeholders"""
    df = pd.DataFrame({
        'item': ['test item']
    })
    prompt_template = "Classify {item} in {nonexistent_column}"
    
    with pytest.raises(TemplateError):
        asyncio.run(classify_all(df, prompt_template, TestResponse))


def test_process_csv(test_files):
    """Test the process_csv function"""
    input_file, output_file = test_files
    
    # Create test data
    df = pd.DataFrame({
        'item': ['item1', 'item2'],
        'category': ['cat1', 'cat2']
    })
    df.to_csv(input_file, index=False)
    
    mock_responses = [
        TestResponse(reason="test reason 1", classification="person"),
        TestResponse(reason="test reason 2", classification="place")
    ]
    
    async def mock_classify_all(*args, **kwargs):
        return mock_responses
    
    with patch('classifier.classify_all', side_effect=mock_classify_all):
        prompt_template = "Classify {item} in {category}"
        process_csv(input_file, output_file, prompt_template, TestResponse)
        
        # Verify the output
        result_df = pd.read_csv(output_file)
        assert len(result_df) == 2
        assert 'reason' in result_df.columns
        assert 'classification' in result_df.columns
        assert result_df['classification'].tolist() == ['person', 'place']


def test_process_csv_preserves_existing(test_files):
    """Test that process_csv preserves existing non-NA classifications"""
    input_file, output_file = test_files
    
    # Create test data with some pre-existing classifications
    df = pd.DataFrame({
        'item': ['a cat', 'a pencil', 'an oil derrick'],
        'reason': ['some reason', 'NA', ''],
        'classification': ['person', 'thing', '']
    })
    df.to_csv(input_file, index=False)
    
    mock_responses = [
        TestResponse(reason="new reason", classification="thing"),  # This should not override existing classification
        TestResponse(reason="new reason 2", classification="place")  # This should be applied
    ]
    
    async def mock_classify_all(*args, **kwargs):
        return mock_responses
    
    with patch('classifier.classify_all', side_effect=mock_classify_all):
        prompt_template = "Classify {item}"
        process_csv(input_file, output_file, prompt_template, TestResponse)
        
        # Verify the output
        result_df = pd.read_csv(output_file)
        assert len(result_df) == 3
        assert result_df['classification'].tolist() == ['person', 'thing', 'place']
        assert result_df.loc[0, 'reason'] == 'some reason'  # Original reason should be preserved
        assert result_df.loc[2, 'reason'] == 'new reason 2'  # New reason should be applied 


def test_process_csv_with_nested_model(test_files):
    """Test that process_csv correctly flattens nested model fields into columns"""
    input_file, output_file = test_files
    
    # Create test data
    df = pd.DataFrame({
        'item': ['test item 1', 'test item 2'],
        'primary': ['', ''],
        'confidence': ['', ''],
        'project_type': ['', ''],
        'justification': ['', ''],
        'evidence': ['', '']
    })
    df.to_csv(input_file, index=False)
    
    mock_responses = [
        NestedClassificationResponse(
            classification=ClassificationDetails(
                primary=PrimaryClassification.TYPE_A,
                confidence=ConfidenceLevel.HIGH,
                project_type=ProjectType.X
            ),
            justification="test justification 1",
            evidence="test evidence 1"
        ),
        NestedClassificationResponse(
            classification=ClassificationDetails(
                primary=PrimaryClassification.TYPE_B,
                confidence=ConfidenceLevel.MEDIUM,
                project_type=ProjectType.Y
            ),
            justification="test justification 2",
            evidence="test evidence 2"
        )
    ]
    
    async def mock_classify_all(*args, **kwargs):
        return mock_responses
    
    with patch('classifier.classify_all', side_effect=mock_classify_all):
        prompt_template = "Classify {item}"
        process_csv(input_file, output_file, prompt_template, NestedClassificationResponse)
        
        # Verify the output
        result_df = pd.read_csv(output_file)
        assert len(result_df) == 2
        
        # Check that nested fields are flattened
        assert 'primary' in result_df.columns
        assert 'confidence' in result_df.columns
        assert 'project_type' in result_df.columns
        assert 'justification' in result_df.columns
        assert 'evidence' in result_df.columns
        assert 'classification' not in result_df.columns
        
        # Verify values
        assert result_df.loc[0, 'primary'] == PrimaryClassification.TYPE_A.value
        assert result_df.loc[0, 'confidence'] == ConfidenceLevel.HIGH.value
        assert result_df.loc[0, 'project_type'] == ProjectType.X.value
        assert result_df.loc[0, 'justification'] == "test justification 1"
        assert result_df.loc[0, 'evidence'] == "test evidence 1" 


def test_process_csv_with_validation_errors(test_files):
    """Test that process_csv handles validation errors and saves partial results"""
    input_file, output_file = test_files
    
    # Create test data
    df = pd.DataFrame({
        'item': ['item1', 'item2', 'item3'],
        'reason': ['', '', ''],
        'classification': ['', '', '']
    })
    df.to_csv(input_file, index=False)
    
    # Mock responses where second item causes validation error
    async def mock_classify_all(*args, **kwargs):
        return [
            TestResponse(reason="reason1", classification="person"),
            None,  # Simulates a failed classification
            TestResponse(reason="reason3", classification="place")
        ]
    
    with patch('classifier.classify_all', side_effect=mock_classify_all):
        prompt_template = "Classify {item}"
        process_csv(input_file, output_file, prompt_template, TestResponse)
        
        # Verify the output
        result_df = pd.read_csv(output_file)
        assert len(result_df) == 3
        
        # Check classifications - handle nan values
        classifications = result_df['classification'].tolist()
        assert classifications[0] == 'person'
        assert pd.isna(classifications[1])  # Check for nan instead of empty string
        assert classifications[2] == 'place'
        
        # Check reasons - handle nan values
        reasons = result_df['reason'].tolist()
        assert reasons[0] == 'reason1'
        assert pd.isna(reasons[1])  # Check for nan instead of empty string
        assert reasons[2] == 'reason3' 