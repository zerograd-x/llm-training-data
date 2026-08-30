from sft_tokenization import (
    ChatPromptFormatter,
    ChatTemplateWrapper,
    CompletionRenderer,
    CompletionTransform,
    DefaultCompletionRenderer,
    DefaultCompletionTransform,
    DefaultPromptRenderer,
    DefaultPromptTransform,
    PackedSequence,
    PackedSequenceLayout,
    PromptRenderer,
    PromptTransform,
    ShortestFirstSequencePacker,
    TemplatePromptRenderer,
    TemplatePromptTransform,
    extract_template_fields,
    load_renderer,
    load_transform,
)


def test_legacy_class_aliases_point_to_new_public_names():
    assert ChatTemplateWrapper is ChatPromptFormatter
    assert PromptTransform is PromptRenderer
    assert CompletionTransform is CompletionRenderer
    assert DefaultPromptTransform is DefaultPromptRenderer
    assert DefaultCompletionTransform is DefaultCompletionRenderer
    assert TemplatePromptTransform is TemplatePromptRenderer
    assert PackedSequenceLayout is PackedSequence
    assert load_transform is load_renderer


def test_new_template_field_name():
    assert extract_template_fields("{title} {brand} {title}") == ["title", "brand"]


def test_new_packer_parameter_names():
    packer = ShortestFirstSequencePacker(token_budget=8, pending_budget_multiplier=2)
    packed = packer.pack([([1, 2], [-100, 2])], pad_id=0)

    assert packed.packed_example_count == 1
    assert packed.padding_length == 6
    assert packer.token_budget == 8
    assert packer.max_pending_tokens == 16
