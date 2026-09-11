"""Shared verifier configuration for experiment runners.

Options are a JSON object keyed by registered rule name, allowing a multi-rule
sweep to configure PAWS without passing its arguments to other verifiers.
"""
import json

from specdiff import available_verifiers, create_verifier


def parse_verifier_options(value):
    options = json.loads(value) if isinstance(value, str) else value
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("verifier options must be an object keyed by rule name")
    for rule, kwargs in options.items():
        if rule not in available_verifiers() or not isinstance(kwargs, dict):
            raise ValueError("verifier options require registered names and option objects")
        create_verifier(rule, **kwargs)  # Fail before allocating models or running a sweep.
    return options


def check_verifier_options(name, value):
    try:
        parse_verifier_options(value)
    except (ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"{name}: {exc}") from exc


def configured_verifier(rule, options=None):
    return create_verifier(rule, **parse_verifier_options(options).get(rule, {}))
