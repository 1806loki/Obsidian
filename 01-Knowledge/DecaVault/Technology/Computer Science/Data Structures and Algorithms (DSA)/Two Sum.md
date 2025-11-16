Given an array of numbers *num* and *target* , Sum of two numbers of the array should be equal to the target

**Brute Force :**
```js
const twoSum = (arr, target) => {
for(let i = 0; i < arr.length ; i++){
for (let j = i +1; j < )
}
}
```


**Optimized Way :**

```js
const twoSum = (arr, target) => {
  let hashMap = {}

  for (let i = 0; i < arr.length; i++) {
    const complementary = target - arr[i]
    if (hashMap.hasOwnProperty(complementary)) {
      return [hashMap[complementary], i]
    }
    hashMap[arr[i]] = i
  }
}
```




