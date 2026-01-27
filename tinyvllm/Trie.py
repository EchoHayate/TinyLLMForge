
class TrieNode:
    """Trie树节点定义"""
    def __init__(self):
        self.children = {}  # 子节点映射: {字符: TrieNode}
        self.is_end = False  # 是否是完整key的结束节点
        self.value = None    # 缓存值（仅is_end=True时有效）


class PrefixCache:
    def __init__(self):
        self.root = TrieNode()  # Trie树根节点

    def add(self, key: str, value) -> None:
        """
        新增缓存项：将key（字符串）和value加入前缀缓存
        :param key: 缓存键（非空字符串）
        :param value: 缓存值（任意类型）
        """
        # 边界校验：禁止空key
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError("key必须是非空字符串")
        
        current_node = self.root
        # 按字符逐个插入Trie树
        for char in key:
            if char not in current_node.children:
                current_node.children[char] = TrieNode()
            current_node = current_node.children[char]
        # 标记为完整key的结束，并存储值
        current_node.is_end = True
        current_node.value = value

    def delete(self, key: str) -> bool:
        """
        删除指定key的缓存项
        :param key: 要删除的缓存键
        :return: 是否删除成功（key存在则返回True，否则False）
        """
        # 递归删除辅助函数：返回当前节点是否需要被清理（子节点为空且非结束节点）
        def _delete(node: TrieNode, key: str, index: int) -> bool:
            # 递归终止：遍历到key最后一个字符
            if index == len(key):
                # 如果不是结束节点，说明key不存在
                if not node.is_end:
                    return False
                # 取消结束标记，清空值
                node.is_end = False
                node.value = None
                # 如果当前节点无子女，需要被父节点清理
                return len(node.children) == 0
            
            char = key[index]
            # 字符不存在，key不存在
            if char not in node.children:
                return False
            
            # 递归删除子节点
            need_remove = _delete(node.children[char], key, index + 1)
            # 如果子节点需要清理，删除当前字符对应的子节点
            if need_remove:
                del node.children[char]
                # 若当前节点无子女且非结束节点，需要被父节点清理
                return len(node.children) == 0 and not node.is_end
            return False

        return _delete(self.root, key, 0)


    def query(self, prefix: str) -> list[tuple[str, any]]:
        """
        查询所有以prefix为前缀的缓存项
        :param prefix: 前缀字符串（空字符串表示查询所有缓存项）
        :return: 列表，每个元素为(匹配的key, 对应value)
        """
        # 第一步：找到prefix最后一个字符对应的Trie节点
        current_node = self.root
        for char in prefix:
            if char not in current_node.children:
                return []  # 前缀不存在，返回空
            current_node = current_node.children[char]
        
        # 第二步：递归收集该节点下所有完整的key-value
        result = []
        def _collect(node: TrieNode, current_key: str):
            # 如果当前节点是完整key的结束，加入结果
            if node.is_end:
                result.append((prefix + current_key, node.value))
            # 遍历所有子节点，递归收集
            for char, child_node in node.children.items():
                _collect(child_node, current_key + char)
        
        _collect(current_node, "")
        return result

    def __str__(self):
        """辅助方法：打印所有缓存项（方便调试）"""
        all_items = self.query("")
        return f"PrefixCache({all_items})"


# ---------------------- 测试案例 ----------------------
if __name__ == "__main__":
    # 1. 初始化缓存
    cache = PrefixCache()

    # 2. 新增缓存项
    cache.add("apple", 10)
    cache.add("app", 5)
    cache.add("banana", 20)
    cache.add("appreciate", 15)
    print("初始缓存:", cache)  # 输出所有项

    # 3. 前缀查询
    print("\n查询前缀'app'的结果:", cache.query("app"))  # 应返回[('app',5), ('apple',10), ('appreciate',15)]
    print("查询前缀'ban'的结果:", cache.query("ban"))    # 应返回[('banana',20)]
    print("查询前缀'pear'的结果:", cache.query("pear"))  # 应返回[]

    # 4. 删除缓存项
    print("\n删除'app'是否成功:", cache.delete("app"))  # True
    print("删除后缓存:", cache)
    print("删除不存在的'pear'是否成功:", cache.delete("pear"))  # False

    # 5. 再次查询前缀'app'
    print("\n删除'app'后查询前缀'app'的结果:", cache.query("app"))  # 应返回[('apple',10), ('appreciate',15)]