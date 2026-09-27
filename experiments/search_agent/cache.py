"""Caches live for one request, including in-flight task coalescing."""
import asyncio
import copy


class RequestCache:
    def __init__(self):
        self.tasks = {}
        self.hits = 0
        self.misses = 0

    async def get(self, key, calculate):
        if key in self.tasks:
            self.hits += 1
        else:
            self.misses += 1
            self.tasks[key] = asyncio.create_task(calculate())
        task = self.tasks[key]
        try:
            return copy.deepcopy(await asyncio.shield(task))
        except BaseException:
            if task.done() and self.tasks.get(key) is task:
                self.tasks.pop(key)
            raise

    async def close(self):
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
