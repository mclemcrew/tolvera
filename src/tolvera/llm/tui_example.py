from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import (
    Header, Footer, Input, Button, TextArea, Select, Label, Log, Static, Collapsible
)
from textual.binding import Binding
import asyncio
import subprocess
import sys
import tempfile
import os
import atexit
import logging

from tolvera.llm import SketchAgent
from tolvera.llm.utils import OllamaModelManager, check_ollama_connection

class SketchGeneratorApp(App):
    CSS = """
    .sidebar {
        width: 35%;
        border: solid $primary;
        margin: 1;
    }
    
    .main-content {
        layout: vertical;
        width: 65%;
        margin: 1;
    }
    
    #content-container {
        display: block;
        width: 100%;
        height: 100%;
    }
    
    #content-container.hidden {
        display: none;
    }
    
    #loading-container {
        align: center middle;
        background: $surface;
        display: none;
        width: 100%;
        height: 100%;
    }
    
    #loading-container.active {
        display: block;
    }
    
    #custom-spinner {
        width: auto;
        text-align: center;
    }

    .code-area {
        height: 55%;
        border: solid $accent;
        margin-bottom: 1;
    }
    
    .output-area {
        height: 35%;
        border: solid $warning;
    }
    
    .generation-controls {
        height: auto;
        margin-bottom: 1;
    }
    
    .model-info {
        background: $boost;
        color: $text;
        margin: 1;
        padding: 1;
        border: solid $secondary;
    }
    
    .status-info {
        background: $surface;
        color: $text;
        margin: 1;
        padding: 1;
        border: solid $secondary;
    }
    """
    
    # Setup for future use??
    BINDINGS = [
        Binding("ctrl+g", "generate", "Generate"),
        Binding("ctrl+r", "run", "Run"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("f1", "help", "Help"),
    ]
    
    def __init__(self):
        # Setup for current Textualize state
        super().__init__()
        self.agent = None
        self.model_manager = OllamaModelManager()
        self.current_config = None
        self.current_code = ""
        self.current_sketch_process = None
        self.temp_sketch_path = None
        self.available_models = []
        self.spinner_timer = None
        self.spinner_frame = 0
        
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Horizontal():
            # sidebar with controls and model selection
            with Vertical(classes="sidebar"):
                yield Label("Tölvera Sketch Generator", classes="model-info")
                
                # model selection dropdown
                with Collapsible(title="Model Selection", collapsed=False):
                    yield Label("Select Model:", classes="status-info")
                    yield Select(
                        options=[("Loading...", "loading")],
                        id="model-select",
                        allow_blank=False
                    )
                    yield Button("Refresh Models", id="refresh-models", variant="default")
                
                # information
                yield Label("Status: Ready", id="status-label", classes="status-info")
            
            # Main content area
            with Vertical(classes="main-content"):
                # The loading overlay, hidden by default, shown when spinner is going
                with Container(id="loading-container"):
                    yield Static("", id="custom-spinner")
                
                with Container(id="content-container"):
                    # Generation controls
                    yield Input(
                        placeholder="Describe your sketch: 'Organic flocking with 2 species leaving slime trails'",
                        id="description-input"
                    )
                    
                    with Horizontal(classes="generation-controls"):
                        yield Button("Generate", variant="primary", id="generate-btn")
                        yield Button("Explain", id="explain-btn")
                        yield Button("Run", variant="success", id="run-btn")
                        yield Button("Stop", variant="error", id="stop-btn")
                    
                    with Vertical(classes="code-area"):
                        yield Label("Generated Code:")
                        yield TextArea(
                            text="# Generated Tölvera sketch will appear here...\n# Use Ctrl+G to generate your first sketch!",
                            language="python",
                            theme="monokai",
                            id="code-display",
                            read_only=False
                        )
                    
                    with Vertical(classes="output-area"):
                        yield Label("Output & Analysis:")
                        yield Log(id="output-log", auto_scroll=True)
        
        yield Footer()
    
    async def on_mount(self) -> None:
        self.title = "Tölvera Sketch Generator"
        
        # Initialize models
        await self.refresh_models()
        
        self.query_one("#description-input").focus()
        self.update_button_states()
        
        await self.check_system_status()

    def log_output(self, message: str, level: str = "info") -> None:
        # Mainly for debugging for me
        log_widget = self.query_one("#output-log", Log)
        prefix = {
            "info": "[INFO]", 
            "success": "[SUCCESS]", 
            "warning": "[WARNING]", 
            "error": "[ERROR]",
            "debug": "[DEBUG]"
        }.get(level, "[LOG]")
        log_widget.write_line(f"{prefix} {message}")
    
    def update_status(self, message: str) -> None:
        status_label = self.query_one("#status-label", Label)
        status_label.update(f"Status: {message}")
    
    def show_spinner(self, message: str = "Working..."):
        try:
            content = self.query_one("#content-container")
            loader = self.query_one("#loading-container")
            
            content.add_class("hidden")
            loader.add_class("active")
            
            self.update_status(message)
            self.start_spinner_animation()
        except Exception as e:
            self.log_output(f"Error showing spinner: {e}", "debug")
    
    def hide_spinner(self):
        try:
            content = self.query_one("#content-container")
            loader = self.query_one("#loading-container")

            loader.remove_class("active")
            content.remove_class("hidden")
            
            self.stop_spinner_animation()
        except Exception as e:
            self.log_output(f"Error hiding spinner: {e}", "debug")

    def start_spinner_animation(self):
        if self.spinner_timer:
            self.spinner_timer.cancel()
        
        self.spinner_frame = 0
        self.animate_spinner()
    
    def stop_spinner_animation(self):
        if self.spinner_timer:
            self.spinner_timer.cancel()
            self.spinner_timer = None
    
    def animate_spinner(self):
        """S/O to Gemini for helping through these animations"""
        try:
            custom_spinner = self.query_one("#custom-spinner", Static)

            c1 = "bright_blue"
            c2 = "bright_red"
            c3 = "bright_green"
            c4 = "yellow"
            c5 = "bright_magenta"
            c6 = "bright_cyan"
            
            patterns = [
                # 1. Swarm
                [
                    f"  [{c1}]•[/] [{c2}]•[/]   [{c1}]•[/] [{c2}]•[/]",
                    f"[{c2}]•[/]   [{c1}]•[/] [{c2}]•[/]   [{c1}]•[/]",
                    f" [{c1}]•[/] [{c2}]•[/]   [{c1}]•[/] [{c2}]•[/] ",
                    f"   [{c2}]•[/] [{c1}]•[/] [{c2}]•[/]   ",
                    f" [{c1}]•[/] [{c2}]•[/] [{c1}]•[/] [{c2}]•[/] [{c1}]•[/] ",
                    f"[{c2}]•[/] [{c1}]•[/]  [{c2}]•[/]  [{c1}]•[/] [{c2}]•[/]",
                    f"  [{c1}]•[/]  [{c2}]•[/]  [{c1}]•[/]  "
                ],
                # 2. Pulsar (Game of Life)
                [
                    f"      [{c3}]●●●[/]   [{c3}]●●●[/]      ",
                    f"  [{c3}]●[/]     [{c3}]●[/] [{c3}]●[/]     [{c3}]●[/]  ",
                    f"[{c3}]●[/]   [{c4}]●●[/]   [{c4}]●●[/]   [{c3}]●[/]",
                    f"[{c3}]●[/]     [{c4}]●[/] [{c4}]●[/]     [{c3}]●[/]",
                    f"  [{c3}]●[/]   [{c4}]●●[/]   [{c4}]●●[/]   [{c3}]●[/]  ",
                    f"      [{c3}]●●●[/]   [{c3}]●●●[/]      "
                ],
                # 3. Growing Slime
                [
                    f"      [{c6}]~[/]      ",
                    f"    [{c6}]~[/] [{c3}]~[/] [{c6}]~[/]    ",
                    f"  [{c6}]~[/] [{c3}]~[/] [{c2}]@ [/] [{c3}]~[/] [{c6}]~[/]  ",
                    f"    [{c3}]~[/] [{c2}]@ [/] [{c2}]@[/] [{c3}]~[/]    ",
                    f"  [{c3}]~[/] [{c2}]@[/] [{c3}]~[/] [{c2}]@[/] [{c3}]~[/]  ",
                    f"    [{c3}]~[/] [{c2}]@[/] [{c3}]~[/]    ",
                    f"      [{c2}]@[/]      "
                ],
                # 4. Neural Network Pulse
                [
                    f"[{c1}](A)[/] --[{c4}]--[/] [{c2}](B)[/]",
                    f"[{c1}](A)[/] ---[{c4}]-[/] [{c2}](B)[/]",
                    f"[{c1}](A)[/] ----[{c4}]-[/] [{c2}](B)[/]",
                    f"[{c1}](A)[/] --[{c4}]--[/] [{c2}](B)[/]",
                    f"[{c1}](A)[/] -[{c4}]---[/] [{c2}](B)[/]",
                    f"[{c1}](A)[/] [{c4}]----[/] [{c2}](B)[/]",
                ],
                 # 5. DNA Helix
                [
                    f"  [{c1}]A[/]----[{c4}]T[/]  ",
                    f" [{c1}]A[/]-[{c6}]G[/]--[{c5}]C[/]-[{c4}]T[/] ",
                    f"   \\ [{c6}]G[/]-[{c5}]C[/] /   ",
                    "    X    ",
                    f"   / [{c5}]C[/]-[{c6}]G[/] \\   ",
                    f" [{c4}]T[/]-[{c5}]C[/]--[{c6}]G[/]-[{c1}]A[/] ",
                    f"  [{c4}]T[/]----[{c1}]A[/]  ",
                ]
            ]
            
            pattern_index = (self.spinner_frame // 10) % len(patterns)
            pattern = patterns[pattern_index]
            frame_in_pattern = self.spinner_frame % len(pattern)

            status_label = self.query_one("#status-label", Label)
            message = status_label.renderable.plain.replace("Status: ", "")

            content = f"[bold]{message}[/bold]\n\n" + pattern[frame_in_pattern]
            
            custom_spinner.update(content)
            
            self.spinner_frame += 1
            
            self.spinner_timer = self.set_timer(0.15, self.animate_spinner)
            
        except Exception:
            # This can fail during shutdown, so we ignore it
            pass
        
    def update_button_states(self):
        try:
            has_code = bool(self.current_code and self.current_code.strip())
            is_running = self.current_sketch_process and self.current_sketch_process.poll() is None
            has_agent = self.agent is not None
            is_generating = False
            
            try:
                loader = self.query_one("#loading-container")
                is_generating = loader.has_class("active")
            except Exception:
                pass

            self.query_one("#explain-btn", Button).disabled = not (has_code and has_agent) or is_generating
            self.query_one("#generate-btn", Button).disabled = not has_agent or is_generating
            
            self.query_one("#run-btn", Button).disabled = is_running or not has_code or is_generating
            self.query_one("#stop-btn", Button).disabled = not is_running
        except Exception:
            # Can happen if widgets aren't mounted yet
            pass
            
    async def refresh_models(self) -> None:
        # Lots of error handling here when booting this up to make sure that ollama is running
        try:
            self.log_output("Refreshing available models...")
            
            if not check_ollama_connection():
                self.log_output("Ollama not running. Please start Ollama first.", "error")
                self.available_models = []
                select_widget = self.query_one("#model-select", Select)
                select_widget.set_options([("No models (Ollama not running)", "none")])
                return
            
            models = self.model_manager.get_available_models(refresh=True)
            
            if not models:
                self.log_output("No models found. Install models with 'ollama pull <model>'", "warning")
                self.available_models = []
                select_widget = self.query_one("#model-select", Select)
                select_widget.set_options([("No models available", "none")])
                return
            
            compatible_models = [m for m in models if self.model_manager.is_model_compatible(m)]
            other_models = [m for m in models if not self.model_manager.is_model_compatible(m)]
            
            options = []
            for model in compatible_models:
                options.append((f"[Compatible] {model}", model))
            for model in other_models:
                options.append((f"[Untested] {model}", model))
            
            self.available_models = models
            select_widget = self.query_one("#model-select", Select)
            select_widget.set_options(options)
            
            # Auto-select best model available
            best_model = self.model_manager.select_best_model()
            if best_model:
                select_widget.value = best_model
                self.log_output(f"Auto-selecting best model: {best_model}")
            else:
                self.log_output("No suitable model found for auto-selection", "warning")
            
            self.log_output(f"Found {len(compatible_models)} compatible models", "success")
            
        except Exception as e:
            self.log_output(f"Error refreshing models: {e}", "error")
    
    async def initialize_agent(self, model_name: str) -> bool:
        try:
            self.show_spinner("Initializing agent...")
            self.log_output(f"Initializing agent with {model_name}...")
            
            if model_name not in self.available_models:
                self.log_output(f"Warning: {model_name} not in available models list", "warning")
            
            await asyncio.sleep(0.1)
            
            self.agent = SketchAgent(model_name=model_name)
            
            if hasattr(self.agent, 'model_name'):
                actual_model = self.agent.model_name
                if actual_model != model_name:
                    self.log_output(f"Warning: Agent using {actual_model} instead of {model_name}", "warning")
                else:
                    self.log_output(f"Agent confirmed using model: {actual_model}", "success")
            
            self.update_status(f"Agent ready ({model_name})")
            self.log_output("Agent initialized successfully", "success")
            return True
            
        except Exception as e:
            self.update_status("Agent initialization failed")
            self.log_output(f"Failed to initialize agent: {e}", "error")
            self.log_output(f"Error details: {str(e)}", "debug")
            self.agent = None
            return False
        finally:
            self.hide_spinner()
            self.update_button_states()
    
    async def check_system_status(self) -> None:
        validation = self.model_manager.validate_setup()
        
        if not validation["ollama_running"]:
            self.log_output("Ollama is not running", "warning")
            self.log_output("Start with: ollama serve", "info")
        elif validation["compatible_models"] == 0:
            self.log_output("No compatible models found", "warning")
            self.log_output("Install with: ollama pull qwen3:4b", "info")
        else:
            self.log_output(f"System ready - {validation['compatible_models']} compatible models", "success")

    async def generate_sketch(self) -> None:
        if not self.agent:
            self.log_output("No agent available. Please select a model first.", "error")
            return
        
        description_input = self.query_one("#description-input", Input)
        description = description_input.value.strip()
        
        if not description:
            self.log_output("Please enter a description", "warning")
            description_input.focus()
            return
        
        self.show_spinner("Generating sketch...")
        self.update_button_states()
        
        try:
            self.log_output(f"Generating sketch from description: '{description}'")
            self.log_output(f"Using model: {getattr(self.agent, 'model_name', 'unknown')}")
            
            self.current_config = None
            self.current_code = ""
            
            await asyncio.sleep(0.1)
            
            # Generate the sketch
            result = await self.agent.generate_sketch(description)
            
            if not result or not result.config or not result.python_code:
                raise ValueError("Generation returned empty or invalid result")
            
            code_display = self.query_one("#code-display", TextArea)
            code_display.text = result.python_code
            self.current_code = result.python_code
            self.current_config = result.config
            
            # Log what was actually generated
            self.log_output(f"Generated sketch: '{result.config.sketch_name}'", "success")
            self.log_output(f"Description: {result.config.description}")
            self.log_output(f"Behaviors: {result.config.get_behavior_summary()}")
            self.log_output(f"Total particles: {result.config.get_total_particle_count()}")
            
            self.update_status("Generation complete")
            self.log_output("Generation complete!", "success")
            self.log_output(f"Explanation: {result.explanation}")
            
            # Show suggestions if available
            if result.suggestions:
                self.log_output("Suggestions:")
                for suggestion in result.suggestions[:3]:
                    self.log_output(f"  {suggestion}")
            
        except Exception as e:
            self.update_status("Generation failed")
            self.log_output(f"Generation failed: {e}", "error")
            self.log_output(f"Error type: {type(e).__name__}", "debug")
        finally:
            # Hide spinner and update buttons
            self.hide_spinner()
            self.update_button_states()
    
    async def explain_sketch(self) -> None:
        if not self.current_code:
            self.log_output("No sketch to explain.", "warning")
            return
        
        # For now, provide basic analysis based on the config
        if self.current_config:
            self.log_output("Sketch Analysis:", "info")
            self.log_output(f"Name: {self.current_config.sketch_name}")
            self.log_output(f"Description: {self.current_config.description}")
            self.log_output(f"Total particles: {self.current_config.get_total_particle_count()}")
            self.log_output(f"Behaviors: {self.current_config.get_behavior_summary()}")
            
            # Analysis of each behavior
            for i, behavior in enumerate(self.current_config.behaviors):
                self.log_output(f"Behavior {i+1}: {behavior.behavior_type.value} ({behavior.particle_count} particles)")
        else:
            self.log_output("Basic code analysis available only", "info")

    async def run_sketch(self) -> None:
        if not self.current_code:
            self.log_output("No sketch to run.", "warning")
            return
        
        await self.stop_sketch()
        
        try:
            # Create temporary file for the sketch
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as temp_f:
                self.temp_sketch_path = temp_f.name
                temp_f.write(self.current_code)

            self.update_status("Launching sketch...")
            self.log_output("Launching sketch...")
            
            self.current_sketch_process = subprocess.Popen(
                [sys.executable, self.temp_sketch_path], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            await asyncio.sleep(1.0)
            
            if self.current_sketch_process.poll() is None:
                self.update_status("Sketch running")
                self.log_output("Sketch launched successfully - check for new window", "success")
            else:
                _, stderr = self.current_sketch_process.communicate()
                error_message = stderr.decode().strip()
                self.update_status("Sketch failed to launch")
                self.log_output(f"Sketch failed to launch: {error_message}", "error")
                self.cleanup_processes()
                
        except Exception as e:
            self.update_status("Launch failed")
            self.log_output(f"Failed to launch sketch: {e}", "error")
            self.cleanup_processes()
        finally:
            self.update_button_states()
    
    async def stop_sketch(self) -> None:
        if self.current_sketch_process and self.current_sketch_process.poll() is None:
            self.update_status("Stopping sketch...")
            self.log_output("Stopping sketch...")
            try:
                self.current_sketch_process.terminate()
                await asyncio.to_thread(self.current_sketch_process.wait, timeout=2.0)
                self.update_status("Sketch stopped")
                self.log_output("Sketch stopped", "success")
            except subprocess.TimeoutExpired:
                self.current_sketch_process.kill()
                self.update_status("Sketch force-stopped")
                self.log_output("Sketch force-killed", "warning")
            except Exception as e:
                self.log_output(f"Error stopping process: {e}", "error")

        self.cleanup_processes()
        self.update_button_states()

    def cleanup_processes(self) -> None:
        if self.current_sketch_process:
            self.current_sketch_process = None
        if self.temp_sketch_path and os.path.exists(self.temp_sketch_path):
            try:
                os.remove(self.temp_sketch_path)
                self.temp_sketch_path = None
            except OSError:
                pass

    # Event handlers
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_actions = {
            "generate-btn": self.generate_sketch,
            "explain-btn": self.explain_sketch,
            "run-btn": self.run_sketch,
            "stop-btn": self.stop_sketch,
            "refresh-models": self.refresh_models,
        }
        
        if event.button.id in button_actions:
            await button_actions[event.button.id]()
    
    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-select" and event.value != "none" and event.value != "loading":
            self.log_output(f"Model selection changed to: {event.value}")
            
            # Clear any existing agent when switching models
            if self.agent:
                self.log_output("Clearing previous agent...")
                self.agent = None
                self.update_button_states()
            
            # Initialize with the newly selected model
            success = await self.initialize_agent(event.value)
            if not success:
                self.log_output(f"Failed to initialize {event.value}, trying fallback...", "warning")
                # Try to fall back to a working model
                fallback = self.model_manager.select_best_model()
                if fallback and fallback != event.value:
                    self.log_output(f"Attempting fallback to {fallback}")
                    await self.initialize_agent(fallback)
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "description-input":
            await self.generate_sketch()
    
    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "code-display":
            self.current_code = event.text_area.text
            self.update_button_states()
    
    async def action_generate(self) -> None:
        await self.generate_sketch()
    
    async def action_run(self) -> None:
        await self.run_sketch()
    
    async def action_help(self) -> None:
        self.log_output("Keyboard Shortcuts:", "info")
        self.log_output("  Ctrl+G: Generate sketch from description", "info")
        self.log_output("  Ctrl+R: Run current sketch", "info")
        self.log_output("  Ctrl+Q: Quit application", "info")
        self.log_output("  F1: Show this help", "info")
    
    async def action_quit(self) -> None:
        await self.stop_sketch()
        self.stop_spinner_animation()
        self.exit()


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = SketchGeneratorApp()
    atexit.register(app.cleanup_processes)
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()