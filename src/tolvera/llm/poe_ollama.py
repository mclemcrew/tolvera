
"""
Ollama integration for PoE expert synthesis.

This module provides the LLM integration for generating Taichi-compatible expert functions from natural language descriptions.
"""

import logging
import re
from typing import Optional, List, Dict, Any, Tuple
import ollama
from .prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class OllamaModelManager:

    def __init__(self):
        self.compatible_models = [
            "qwen2.5:3b", "qwen2.5:7b", "qwen2.5-coder:7b",
            "gemma2:2b", "gemma2:9b",
            "qwen2.5:3b", "llama3.2:1b",
            "mistral:7b", "mistral-nemo:12b"
        ]
        self.default_model = "qwen2.5:3b"
        try:
            self.client = ollama.Client()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Ollama client. Is Ollama running? Error: {e}")

    def check_ollama_running(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def list_available_models(self) -> List[str]:
        try:
            models_info = self.client.list()
            return [model['name'] for model in models_info.get('models', [])]
        except Exception:
            return []

    def ensure_compatible_model(
            self, requested_model: Optional[str] = None) -> str:
        available = self.list_available_models()

        if requested_model and requested_model in available:
            logger.info(f"Using requested model: {requested_model}")
            return requested_model

        for model in self.compatible_models:
            if model in available:
                logger.info(f"Using compatible model: {model}")
                return model

        logger.warning(f"No compatible model found. Available: {available}")
        logger.info(f"Pulling default model: {self.default_model}")

        try:
            ollama.pull(self.default_model)
            return self.default_model
        except Exception as e:
            raise RuntimeError(f"Failed to pull default model: {e}")


class OllamaClient:

    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        self.model_manager = OllamaModelManager()

        if not self.model_manager.check_ollama_running():
            raise RuntimeError(
                "Ollama is not running. Please start it with 'ollama serve'")

        self.model_name = self.model_manager.ensure_compatible_model(
            model_name)
        self.client = ollama.AsyncClient()
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")

    async def chat(self,
                   messages: List[Dict[str,
                                       str]],
                   temperature: float = 0.7,
                   max_tokens: int = 2000,
                   think: bool = False) -> str:
        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                },
                # think=think
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")


class PoEExpertSynthesizer:

    def __init__(self, model_name: Optional[str] = None):
        self.client = OllamaClient(model_name)

    def extract_code(self, response: str) -> str:
        # Try to find code between triple backticks
        code_match = re.search(
            r'```(?:python)?\n(.*?)```',
            response,
            re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Try to find @ti.func definition directly
            func_match = re.search(
                r'(@ti\.func.*?)(?=\n@|\n\n|\Z)',
                response,
                re.DOTALL)
            if func_match:
                code = func_match.group(1).strip()
            else:
                # Return cleaned response
                code = response.strip()

        # Fix common import errors
        code = self._fix_common_import_errors(code)
        return code

    # TODO: This shouldn't have to exist
    def extract_kernel_code(self, response: str) -> str:
        code = self.extract_code(response)

        # For kernel code, find only the @ti.kernel function (because LLM was generating a lot all over?)
        kernel_match = re.search(
            r'(@ti\.kernel\s*\n\s*def\s+apply_all_experts.*?)(?=\n(?:@|def\s+\w+|#\s*Example|$))',
            code,
            re.DOTALL)


        # TODO: This is a mess.  Please fix.
        if kernel_match:
            kernel_code = kernel_match.group(1).strip()
            # Make sure we include the complete function body
            # Count braces/indentation to find the end
            lines = kernel_code.split('\n')
            if lines:
                # Find the base indentation of the def line
                def_line_idx = next(
                    (i for i, line in enumerate(lines) if 'def' in line), 0)
                if def_line_idx < len(lines):
                    def_indent = len(lines[def_line_idx]) - \
                        len(lines[def_line_idx].lstrip())

                    complete_lines = []
                    for i, line in enumerate(lines):
                        complete_lines.append(line)
                        if i > def_line_idx and line.strip() and len(
                                line) - len(line.lstrip()) <= def_indent:

                            complete_lines.pop() 
                            break

                    kernel_code = '\n'.join(complete_lines)

            return kernel_code

        # If no kernel found, return the original code
        return code

    # TODO: Fix this again.  All due to poor prompting on my part
    def _fix_common_import_errors(self, code: str) -> str:
        """Fix common import errors in generated code."""
        code = re.sub(r'import\s+ti\.experimental.*?\n', '', code)
        code = re.sub(r'^import\s+ti\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'from\s+ti\s+import.*?\n', '', code)
        lines = code.split('\n')
        seen_imports = set()
        cleaned_lines = []

        for line in lines:
            if not line.strip() and len(
                    cleaned_lines) > 0 and not cleaned_lines[-1].strip():
                continue
            if re.match(r'^\s*import\s+taichi\s+as\s+ti\s*$', line):
                if 'taichi' not in seen_imports:
                    seen_imports.add('taichi')
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def validate_expert_code(self, code: str) -> Tuple[bool, List[str]]:
        errors = []

        if "@ti.func" not in code:
            errors.append("Missing @ti.func decorator")

        if "return" not in code:
            errors.append("Missing return statement for force vector")

        unsafe_patterns = [
            (r'exec\s*\(', "exec() is not allowed"),
            (r'eval\s*\(', "eval() is not allowed"),
            (r'__import__', "__import__ is not allowed"),
            (r'open\s*\(', "file operations not allowed"),
            (r'subprocess', "subprocess operations not allowed")
        ]

        for pattern, message in unsafe_patterns:
            if re.search(pattern, code):
                errors.append(message)

        has_ti_math_alias = bool(
            re.search(
                r'import\s+ti\.math\s+as\s+math',
                code))

        if has_ti_math_alias:
            if re.search(r'^import\s+math\s*$', code, re.MULTILINE):
                errors.append(
                    "Conflicting math imports: already importing ti.math as math")
        else:
            if re.search(r'(?<!ti\.)math\.(sqrt|sin|cos|tan)', code):
                errors.append(
                    "Use ti.sqrt, ti.sin, ti.cos instead of Python math")

        return len(errors) == 0, errors

    # This one feels alright for now
    async def analyze_description(self, description: str) -> str:
        """Use LLM router to analyze description and determine implementation category."""
        logger.info(f"Analyzing description: '{description}'")

        system_prompt = load_prompt("router_analysis_system")
        user_prompt = load_prompt("router_analysis_user").format(
            description=description)

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await self.client.chat(messages, temperature=0.1, think=False)

        valid_categories = [
            "FORCE_ONLY", # I don't like this, but wanted to separate concerns right now
            "STATE_TRACKING",
            "PARTICLE_INTERACTION"]
        category = None

        cleaned_response = response.strip().upper()
        for line in cleaned_response.split('\n'):
            line = line.strip()
            if line in valid_categories:
                category = line
                break

        # If no valid category found by lines, try the whole response
        if not category:
            # Remove all whitespace and newlines
            compact_response = ''.join(cleaned_response.split())
            for valid_cat in valid_categories:
                if valid_cat in compact_response:
                    category = valid_cat
                    break

        logger.info(
            f"Router categorized as: {category} (from response: '{response.strip()}')")

        if category not in valid_categories:
            raise ValueError(
                f"Invalid category '{category}'. Expected one of: {valid_categories}")

        return category

    # TODO: Move this to synthesize
    async def synthesize_state_from_description(
            self, description: str, category: str) -> Dict[str, Any]:
        logger.info(
            f"Synthesizing state for: '{description}' (category: {category})")

        if category == "PARTICLE_INTERACTION":
            state_type = "per_particle"
            interaction_type = "particle_interaction"
        elif category == "STATE_TRACKING":
            state_type = "per_particle"
            interaction_type = "time_based"
        else:
            raise ValueError(f"Category {category} does not require state")

        system_prompt = load_prompt("expert_state_synthesis_system")
        user_prompt = load_prompt("expert_state_synthesis_user").format(
            description=description,
            state_type=state_type,
            interaction_type=interaction_type)

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await self.client.chat(messages, temperature=0.5, think=False)
        logger.debug(f"Raw state response:\n{response}")

        code = self.extract_code(response)

        state_name_match = re.search(r'tv\.s\.(\w+)\s*=', code)
        if not state_name_match:
            raise ValueError(
                "Could not extract state name from generated code")

        state_name = state_name_match.group(1)

        return {
            "code": code,
            "state_name": state_name,
            "state_type": state_type
        }

    async def synthesize_expert(self,
                                description: str,
                                category: str,
                                state_info: Optional[Dict[str,
                                                          Any]] = None) -> Dict[str,
                                                                                Any]:

        logger.info(
            f"Synthesizing expert for '{description}' (category: {category})")

        if category == "FORCE_ONLY":
            system_prompt = load_prompt("expert_synthesis_system")
            user_prompt = load_prompt("expert_synthesis_user").format(
                description=description)
        elif category in ["PARTICLE_INTERACTION", "STATE_TRACKING"]:
            system_prompt = load_prompt("expert_interaction_synthesis_system")

            state_details = ""
            if state_info:
                state_name = state_info.get('state_name', 'unknown_state')
                state_code = state_info.get('code', '')
                state_fields_match = re.search(
                    r'"state":\s*{\s*([^}]+)\s*}', state_code)

                if state_fields_match:
                    fields_str = state_fields_match.group(1)
                    field_names = re.findall(r'"(\w+)":',
                                             fields_str)
                    state_details = f"""The following state is available for you to use:
- State Name: `tv.s.{state_name}`
- Fields: {field_names}

Example access:
`tv.s.{state_name}.field[i].{field_names[0]}`
""" if field_names else ""

            user_prompt = load_prompt("expert_interaction_synthesis_user").format(
                description=description,
                interaction_type="particle_interaction" if category == "PARTICLE_INTERACTION" else "time_based",
                state_details=state_details)
        else:
            raise ValueError(f"Unknown category: {category}")

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await self.client.chat(messages, temperature=0.5, think=False)
        logger.debug(f"Raw expert response:\n{response}")

        code = self.extract_code(response)

        # Validate code
        is_valid, errors = self.validate_expert_code(code)
        if not is_valid:
            raise ValueError(f"Invalid expert code: {errors}")

        # Extract function name
        name_match = re.search(r'def\s+(\w+)', code)
        if not name_match:
            raise ValueError(
                "Could not extract function name from generated code")

        return {
            "name": name_match.group(1),
            "code": code,
            "expert_type": "force" if category == "FORCE_ONLY" else "interaction"}

    # TODO: Move to synthesizer (this is not going to work.  NEED TO BREAK THIS DOWN!)
    async def synthesize_integration_kernel(
            self,
            expert_info: list,
            state_definitions: dict = None) -> dict:
        logger.info(
            f"Synthesizing integration kernel for {len(expert_info)} experts")

        force_experts = [
            e for e in expert_info if e.get(
                'expert_type',
                'force') == 'force']
        interaction_experts = [
            e for e in expert_info if e.get(
                'expert_type',
                'force') == 'interaction']

        logger.info(f"Force experts: {[e['name'] for e in force_experts]}")
        logger.info(
            f"Interaction experts: {[e['name'] for e in interaction_experts]}")

        if not interaction_experts:
            expert_calls = [
                f"            total_force += {expert['name']}(pos, vel, mass) * {expert['weight']:.2f}"
                for expert in force_experts
            ]
            expert_calls_str = "\n".join(expert_calls)

            system_prompt = load_prompt("kernel_integration_system")
            user_prompt = load_prompt("kernel_integration_user").format(
                expert_calls_str=expert_calls_str)
        else:
            # This is wild and simply doesn't work.
            force_calls = [
                f"total_force += {expert['name']}(pos, vel, mass) * {expert['weight']:.2f}"
                for expert in force_experts
            ]
            interaction_calls = [
                f"total_force += {expert['name']}(i) * {expert['weight']:.2f}"
                for expert in interaction_experts
            ]

            # Build simplified state info
            state_info_lines = []
            if state_definitions:
                for state_name, state_def in state_definitions.items():
                    # Get state fields from the original state definition
                    state_code = state_def.get('code', '')
                    state_dict_match = re.search(
                        r'"state":\s*{([^}]+)}', state_code)

                    if state_dict_match:
                        # Parse the state fields from the code
                        fields_str = state_dict_match.group(1)
                        field_matches = re.findall(
                            r'"(\w+)":\s*\([^)]+\)', fields_str)

                        # TODO: Was trying to fix up generated qwen code from a collision thing
                        for field_name in field_matches:
                            if 'collid' in field_name.lower() or field_name == 'is_collided':
                                state_info_lines.append(
                                    f"State field: tv.s.{state_name}.field[i].{field_name}")

            state_instructions = "\n".join(
                state_info_lines) if state_info_lines else "No collision states defined"

            system_prompt = load_prompt("kernel_integration_full_system")
            user_prompt = load_prompt("kernel_integration_full_user").format(
                force_expert_calls="                " +
                "\n                ".join(force_calls) if force_calls else "# No force experts",
                interaction_expert_calls="                " +
                "\n                ".join(interaction_calls) if interaction_calls else "# No interaction experts",
                state_info=state_instructions)

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = await self.client.chat(messages, temperature=0.0, think=False)
        logger.debug(f"Raw kernel response:\n{response}")

        code = self.extract_kernel_code(response)

        # Validate kernel code
        is_valid, errors = self._validate_kernel_code(code)
        if not is_valid:
            raise ValueError(f"Invalid kernel code: {errors}")

        # Extract function name
        name_match = re.search(r'def\s+(\w+)', code)
        if not name_match:
            raise ValueError("Could not extract kernel function name")

        return {
            "name": name_match.group(1),
            "code": code
        }

    def _validate_kernel_code(self, code: str) -> tuple:
        errors = []

        if "@ti.kernel" not in code:
            errors.append("Missing @ti.kernel decorator")

        if "def " not in code:
            errors.append("Missing function definition")

        function_defs = re.findall(r'^def\s+(\w+)', code, re.MULTILINE)
        if len(function_defs) > 1:
            errors.append(
                f"Multiple function definitions found: {function_defs}. Kernel should only contain apply_all_experts()")

        if re.search(r'#\s*Example of', code, re.IGNORECASE):
            errors.append("Contains example or placeholder code")

        # Check if there's code after the kernel function
        if re.search(r'^\S.*', code.split('\n')
                     [-1]) and not code.strip().endswith(')'):
            errors.append("Code found after kernel function")

        unsafe_patterns = [
            (r'exec\s*\(', "exec() is not allowed"),
            (r'eval\s*\(', "eval() is not allowed"),
            (r'__import__', "__import__ is not allowed"),
            (r'open\s*\(', "file operations not allowed"),
            (r'subprocess', "subprocess operations not allowed")
        ]

        for pattern, message in unsafe_patterns:
            if re.search(pattern, code):
                errors.append(message)

        return len(errors) == 0, errors
