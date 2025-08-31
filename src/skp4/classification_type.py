'classification type'

import logging
import collections
from enum import Enum, auto
from itertools import chain

from skp4.exceptions import SklearnP4Exception

log = logging.getLogger(__name__)

class ClassificationType(Enum):
    BINARY = auto()
    MULTI_CLASS = auto()
    MULTI_LABEL = auto()

def is_iterable(input_iterable):
    'true if any iterable, but not string'
    return isinstance(input_iterable, collections.abc.Iterable) and not isinstance(input_iterable, str)

def get_number_of_nested_labels(nested_input_iterable):
    flatten_list = chain.from_iterable(nested_input_iterable)
    return len(set(flatten_list))

def is_nested_list_format(input_iterable):
    nested_flags = [is_iterable(i) for i in input_iterable]

    any_nested = any(nested_flags)
    all_nested = all(nested_flags)

    if any_nested and not all_nested:
        raise SklearnP4Exception('inconsistent structure of input true/predicted format - mixed flat and nested')

    if all_nested:
        if get_number_of_nested_labels(input_iterable) < 2:
            raise SklearnP4Exception('at least 2 labels required for multi-label classifier')
        return True
    else:
        return False

def get_vector_classification_type(y_vec) -> ClassificationType:
    
    if not is_iterable(y_vec):
        raise SklearnP4Exception('invalid input true/predicted format - not iterable')
    
    if is_nested_list_format(y_vec):
        return ClassificationType.MULTI_LABEL
    elif len(set(y_vec)) > 2:
        return ClassificationType.MULTI_CLASS
    elif len(set(y_vec)) == 2:
        return ClassificationType.BINARY
    else:
        raise SklearnP4Exception('number of classes should be at least 2')


def get_classification_type(y_true, y_pred) -> ClassificationType:
    if len(y_true) != len(y_pred):
        raise SklearnP4Exception('y_true and y_pred should have the same length')
    
    if len(y_true) == 0:
        raise SklearnP4Exception('length of y_true is zero')

    ctype = get_vector_classification_type(y_true)
    log.debug(f'classification type: {ctype.name}')
    return ctype