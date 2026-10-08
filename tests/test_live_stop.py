import asyncio
from unittest.mock import AsyncMock


def test_kis_entrypoints_block_even_approved_order():
    from infra.kis_broker import KISBroker
    broker=object.__new__(KISBroker)
    broker._get_session=AsyncMock(side_effect=AssertionError('network forbidden'))
    for name,args in [('kr_buy',('005930',1)),('kr_sell',('005930',1)),('us_buy',('AAPL',1)),('us_sell',('AAPL',1)),('bond_buy',('BOND',1,100)),('futures_buy',('FUTURE',1,100)),('place_order',(object(),))]:
        assert '정지' in asyncio.run(getattr(broker,name)(*args))
    broker._get_session.assert_not_called()
