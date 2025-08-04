from typing import Generic, TypeVar

class Vocabulary:
    r"""An abstract base class that represents a mapping between token types and
    integer IDs.
    """

    def __len__(self) -> int:
        r"""Get the number of token type-integer ID pairs in this vocabulary.
        """
        raise NotImplementedError

V = TypeVar('V')

class VocabularyBuilder(Generic[V]):
    r"""An abstract base class that can be used for constructing
    :py:class:`Vocabulary` objects of a certain type.
    """

    def content(self, tokens: list[str]) -> V:
        r"""Build a vocabulary that assigns consecutive integer IDs to a list of
        token strings. These are "content" tokens in the sense that they come
        from a corpus and are not special tokens.

        :param tokens: A list of token type strings.
        :return: A vocabulary containing the specified tokens.
        """
        raise NotImplementedError

    def catchall(self, token: str) -> V:
        r"""Build a vocabulary that maps all token types to a single token type.
        This implements the behavior of an UNK token.

        :param token: A token string used to represent the catchall token.
        :return: A vocabulary that maps all token strings to the specified
            token.
        """
        raise NotImplementedError

    def reserved(self, tokens: list[str]) -> V:
        r"""Build a vocabulary that assigns consecutive integer IDs to a list of
        special reserved tokens. The token strings of these special tokens are
        for display purposes only and will never conflict with content tokens.

        :param tokens: A list of token type strings.
        :return: A vocabulary containing the specified tokens.
        """
        raise NotImplementedError
