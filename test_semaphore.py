import asyncio
async def test():
    sem = asyncio.Semaphore(10)
    async def run(i):
        async with sem:
            return i
    tasks = [run(i) for i in range(20)]
    return await asyncio.gather(*tasks)
print(asyncio.run(test()))
