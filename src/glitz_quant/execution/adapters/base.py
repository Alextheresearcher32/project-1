"""
Adapter base class. All execution venues implement this interface.
The OMS only talks to adapters via this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from glitz_quant.data.types import Fill, Order, Venue


class ExchangeAdapter(ABC):
    """One adapter instance per venue."""

    venue: Venue

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """
        Submit `order`. Mutates and returns it with venue_order_id set,
        status updated (SUBMITTED at minimum), or REJECTED with reason.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order: Order) -> bool: ...

    @abstractmethod
    async def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all open orders, optionally filtered by symbol. Return count."""
        ...

    @abstractmethod
    async def poll_order(self, order: Order) -> Order:
        """Refresh order state from the venue."""
        ...

    @abstractmethod
    async def poll_fills(self, order: Order) -> list[Fill]:
        """Return any fills for this order since last poll."""
        ...
