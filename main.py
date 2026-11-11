TITLES: dict[str, tuple[str, list[str]]] = {}



def _add(kind, keywords, *names):
    for n in names:
        TITLES[n] = (kind, keywords)


_add("noble", ["lord", "baron"], "lord")
_add("noble", ["princess", "prinses"], "prinses", "princesse", "princess")

# if __name__ == "__main__":
#     main()
print(TITLES)