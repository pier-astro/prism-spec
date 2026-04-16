"""
Backward-compatible re-export of ``prism.modeling.operators.instrument``.

All classes and functions are defined in the operators subpackage.
Import from ``prism.modeling.operators.instrument`` for new code.
"""
from .operators.instrument import *          # noqa: F401,F403
from .operators.instrument import (          # explicit for IDE support
    InstrumentResponse,
    SpectralResponse,
    ResponseOperator,
    load_responses_mapping,
    add_response_to_archive,
    list_instruments,
    _load_user_mapping,
    _save_user_mapping,
)
from .operators.convolved import (
    LinearOperatorCompoundModel,
    MatrixLinearOperator,
)
