import pytest
import pandas as pd
from pydantic import BaseModel
from llm_classifier.classifier import (
    extract_json_from_markdown,
    get_format_args,
    prepare_dataframe,
    get_empty_mask,
    update_classifications,
    TemplateError
)


class TestModel(BaseModel):
    field1: str
    field2: str


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'col1': ['value1', 'value2'],
        'col2': ['value3', 'value4']
    })


def test_extract_json_from_markdown_with_fence():
    """Test extracting JSON from markdown code fence."""
    content = '```json\n{"key": "value"}\n```'
    result = extract_json_from_markdown(content)
    assert result == '{"key": "value"}'


def test_extract_json_from_markdown_without_fence():
    """Test handling plain JSON without markdown fence."""
    content = '{"key": "value"}'
    result = extract_json_from_markdown(content)
    assert result == '{"key": "value"}'


def test_extract_json_from_markdown_with_quotes():
    """Test handling JSON with surrounding quotes."""
    content = '"{"key": "value"}"'
    result = extract_json_from_markdown(content)
    assert result == '{"key": "value"}'


def test_get_format_args_valid():
    """Test extracting format arguments with valid placeholders."""
    row = pd.Series({'name': 'John', 'age': '30'})
    placeholders = ['name', 'age']
    result = get_format_args(row, placeholders)
    assert result == {'name': 'John', 'age': '30'}


def test_get_format_args_missing_column():
    """Test handling missing column in format arguments."""
    row = pd.Series({'name': 'John'})
    placeholders = ['name', 'nonexistent']
    with pytest.raises(TemplateError, match="Column 'nonexistent' in prompt template not found in CSV file"):
        get_format_args(row, placeholders)


def test_prepare_dataframe_new_columns():
    """Test preparing DataFrame with new columns."""
    df = pd.DataFrame({'existing': ['value']})
    model_fields = ['existing', 'new_field']
    result = prepare_dataframe(df, model_fields)
    
    assert 'new_field' in result.columns
    assert result['new_field'].iloc[0] == ''
    assert result['existing'].iloc[0] == 'value'


def test_prepare_dataframe_all_columns_exist():
    """Test preparing DataFrame when all columns already exist."""
    df = pd.DataFrame({'field1': ['value1'], 'field2': ['value2']})
    model_fields = ['field1', 'field2']
    result = prepare_dataframe(df, model_fields)
    
    assert list(result.columns) == ['field1', 'field2']
    assert result.equals(df)


def test_get_empty_mask_all_types():
    """Test empty mask creation with various empty value types."""
    df = pd.DataFrame({
        'field1': ['value', '', 'NA', None],
        'field2': ['value', 'na', '', pd.NA]
    })
    model_fields = ['field1', 'field2']
    mask = get_empty_mask(df, model_fields)
    
    assert mask.tolist() == [False, True, True, True]


def test_get_empty_mask_no_empty():
    """Test empty mask when no empty values exist."""
    df = pd.DataFrame({
        'field1': ['value1', 'value2'],
        'field2': ['value3', 'value4']
    })
    model_fields = ['field1', 'field2']
    mask = get_empty_mask(df, model_fields)
    
    assert not any(mask)


def test_update_classifications_with_results():
    """Test updating classifications with new results."""
    df = pd.DataFrame({
        'field1': ['keep', '', 'keep'],
        'field2': ['keep', '', 'keep']
    })
    mask = pd.Series([False, True, False])
    results = [TestModel(field1='new1', field2='new2')]
    model_fields = ['field1', 'field2']
    
    result = update_classifications(df, results, mask, model_fields)
    
    assert result.loc[1, 'field1'] == 'new1'
    assert result.loc[1, 'field2'] == 'new2'
    assert result.loc[0, 'field1'] == 'keep'
    assert result.loc[2, 'field1'] == 'keep'


def test_update_classifications_no_updates():
    """Test update_classifications when no updates are needed."""
    df = pd.DataFrame({
        'field1': ['keep', 'keep'],
        'field2': ['keep', 'keep']
    })
    mask = pd.Series([False, False])
    results = []
    model_fields = ['field1', 'field2']
    
    result = update_classifications(df, results, mask, model_fields)
    
    assert result.equals(df)


def test_update_classifications_all_rows():
    """Test updating all rows with new classifications."""
    df = pd.DataFrame({
        'field1': ['', ''],
        'field2': ['', '']
    })
    mask = pd.Series([True, True])
    results = [
        TestModel(field1='new1', field2='new2'),
        TestModel(field1='new3', field2='new4')
    ]
    model_fields = ['field1', 'field2']
    
    result = update_classifications(df, results, mask, model_fields)
    
    assert result['field1'].tolist() == ['new1', 'new3']
    assert result['field2'].tolist() == ['new2', 'new4'] 