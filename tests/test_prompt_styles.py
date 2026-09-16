"""The verbosity control must not disturb the main grid's prompt parity."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import generate


Q = 'Who received the first Nobel Prize in Physics?'
CHUNKS = ['Wilhelm Roentgen received it in 1901.', 'The prize is Swedish.']


class TestPromptStyles:

    def test_default_is_the_standard_template(self):
        """The default must stay byte-identical: every generator in the main
        grid sends it, and H3 compares rankings across those generators."""
        ctx = '[Chunk 1]: %s\n\n[Chunk 2]: %s' % (CHUNKS[0], CHUNKS[1])
        assert generate.build_prompt(Q, CHUNKS) == \
            generate.RAG_PROMPT_TEMPLATE.format(context=ctx, question=Q)

    def test_explicit_standard_equals_default(self):
        assert generate.build_prompt(Q, CHUNKS) == \
            generate.build_prompt(Q, CHUNKS, style='standard')

    def test_verbose_differs_by_exactly_one_line(self):
        """The control varies answer structure and nothing else. If this test
        starts failing, the control is no longer isolating verbosity."""
        std = generate.build_prompt(Q, CHUNKS).split('\n')
        vrb = generate.build_prompt(Q, CHUNKS, style='verbose').split('\n')
        added = [l for l in vrb if l not in std]
        removed = [l for l in std if l not in vrb]
        assert len(added) == 1, added
        assert removed == [], removed

    def test_verbose_keeps_the_grounding_and_refusal_wording(self):
        """`is_abstention` matches on the refusal phrase the prompt offers, and
        the grounding constraint is what faithfulness is measured against. Both
        must survive verbatim or the arm is not comparable."""
        vrb = generate.build_prompt(Q, CHUNKS, style='verbose')
        assert 'I cannot answer based on the provided context.' in vrb
        assert 'using ONLY the provided context' in vrb
        assert 'Do not use any external knowledge.' in vrb

    def test_unknown_style_is_an_error_not_a_silent_fallback(self):
        import pytest
        with pytest.raises(KeyError):
            generate.build_prompt(Q, CHUNKS, style='chatty')
