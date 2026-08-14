"""
Tests for the Q&A Pipeline.

Run: pytest tests/ -v
"""

import os
from unittest.mock import patch

import pytest

from src.knowledge_base import build_knowledge_base
from src.pipeline import ask_question, get_llm, main

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def vector_store():
    """Build the vector store once for all tests."""
    return build_knowledge_base(DATA_DIR)


@pytest.fixture(scope="module")
def llm():
    """Load the LLM once for all tests."""
    return get_llm()


# ────────────────────────────────
# ask_question return structure
# ────────────────────────────────
class TestAskQuestionStructure:
    def test_returns_dict(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result, dict), "ask_question should return a dict"

    def test_has_answer_key(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert "answer" in result, "Result dict must have an 'answer' key"

    def test_has_sources_key(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert "sources" in result, "Result dict must have a 'sources' key"

    def test_answer_is_string(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result["answer"], str), "'answer' should be a string"
        assert len(result["answer"].strip()) > 0, "'answer' should not be empty"

    def test_sources_is_list(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result["sources"], list), "'sources' should be a list"
        assert len(result["sources"]) > 0, "'sources' should not be empty"


# ────────────────────────────────
# Retrieval quality
# ────────────────────────────────
class TestRetrieval:
    def test_retrieves_pricing_info(self, vector_store, llm):
        result = ask_question(vector_store, llm, "How much does the Growth package cost?")
        sources_text = " ".join(result["sources"]).lower()
        assert "growth" in sources_text or "$5,500" in sources_text, (
            "Sources should contain pricing-related content"
        )

    def test_retrieves_seo_info(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Do you offer SEO services?")
        sources_text = " ".join(result["sources"]).lower()
        assert "seo" in sources_text or "keyword" in sources_text, (
            "Sources should contain SEO-related content"
        )

    def test_different_questions_get_different_sources(self, vector_store, llm):
        r1 = ask_question(vector_store, llm, "How does onboarding work?")
        r2 = ask_question(vector_store, llm, "What are your PPC management fees?")
        assert r1["sources"] != r2["sources"], (
            "Different questions should retrieve different chunks"
        )


# ────────────────────────────────
# Answer generation
# ────────────────────────────────
class TestAnswerGeneration:
    def test_answer_is_not_just_the_prompt(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Can I cancel my contract?")
        assert "Context:" not in result["answer"], (
            "Answer should be the generated text, not the full prompt"
        )

    def test_answer_responds_to_question(self, vector_store, llm):
        result = ask_question(vector_store, llm, "How much is the Starter package?")
        answer = result["answer"].lower()
        assert "2,500" in answer or "2500" in answer or "starter" in answer, (
            "Answer should address the pricing question"
        )


class TestAskQuestionExtras:
    def test_returns_three_sources(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert len(result["sources"]) == 3

    def test_sources_are_strings(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What is your onboarding process?")
        assert all(isinstance(source, str) and source.strip() for source in result["sources"])

    def test_empty_question_does_not_search(self, vector_store, llm):
        result = ask_question(vector_store, llm, "   ")
        assert result["answer"] == "Please enter a question."
        assert result["sources"] == []

    def test_retrieves_cancellation_policy(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Can I cancel early?")
        sources_text = " ".join(result["sources"]).lower()
        assert "cancel" in sources_text or "termination" in sources_text

    def test_out_of_scope_question_still_returns_structure(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What is the capital of France?")
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) == 3


class TestCLI:
    def test_quit_exits_cleanly(self, vector_store, llm):
        with patch("src.pipeline.build_knowledge_base", return_value=vector_store), \
             patch("src.pipeline.get_llm", return_value=llm), \
             patch("builtins.input", return_value="quit"):
            main([])

    def test_empty_input_then_quit(self, vector_store, llm, capsys):
        with patch("src.pipeline.build_knowledge_base", return_value=vector_store), \
             patch("src.pipeline.get_llm", return_value=llm), \
             patch("builtins.input", side_effect=["", "quit"]):
            main([])
        captured = capsys.readouterr()
        assert "Please enter a question" in captured.out

    def test_query_flag_prints_answer(self, vector_store, llm, capsys):
        with patch("src.pipeline.build_knowledge_base", return_value=vector_store), \
             patch("src.pipeline.get_llm", return_value=llm):
            main(["--query", "How much is the Starter package?"])
        captured = capsys.readouterr()
        assert "Answer:" in captured.out
        assert "Sources:" in captured.out

    def test_empty_query_flag_exits(self, vector_store, llm):
        with patch("src.pipeline.build_knowledge_base", return_value=vector_store), \
             patch("src.pipeline.get_llm", return_value=llm):
            with pytest.raises(SystemExit) as exc:
                main(["--query", "   "])
        assert exc.value.code == 1

    def test_missing_data_dir_exits(self):
        with pytest.raises(SystemExit) as exc:
            main(["--data-dir", os.path.join("definitely", "not", "a", "real", "dir")])
        assert exc.value.code == 1

    def test_data_dir_with_no_txt_files_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            main(["--data-dir", str(tmp_path)])
        assert exc.value.code == 1

