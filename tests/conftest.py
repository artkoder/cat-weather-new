import asyncio
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: run test in event loop")

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if asyncio.iscoroutinefunction(pyfuncitem.obj):
        loop = pyfuncitem.funcargs.get('event_loop') or asyncio.get_event_loop()
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        loop.run_until_complete(pyfuncitem.obj(**kwargs))
        return True
    return None
