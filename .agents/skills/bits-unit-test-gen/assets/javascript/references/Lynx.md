# Lynx / ReactLynx Unit Test Quick Reference

Lynx is not a browser environment. Prefer the project's existing `@byted-lynx/react/testing-library` style, and mock the Lynx global object and NativeModules.

## Elements and Events

Common built-in elements: `view`, `text`, `image`, `scroll-view`, `list`, `input`. Do not assume HTML tag semantics.

Common events include `bindtap`, `bindinput`, `bindscroll`. Event object shape differs from DOM events; construct only the fields the component reads.

## Basic Structure

```ts
import { render, screen, fireEvent } from '@byted-lynx/react/testing-library';

it('handles tap', async () => {
  render(<Button onClick={onClick} />);
  fireEvent.tap(screen.getByText('Submit'));
  expect(onClick).toHaveBeenCalled();
});
```

If the project does not use this library, follow existing Lynx test utilities. Do not introduce a new framework.

## Required Mocks

| Dependency | Handling |
|---|---|
| `lynx` global | Mock `getSystemInfoSync`, `createSelectorQuery`, storage, event bus |
| NativeModules | Mock methods actually used by the tested component |
| InitData | Provide stable test data |
| timers / animation | Use fake timers or direct mocks |
| bridge / network | Mock to controlled Promises |

## Hooks and Threads

- Test regular React hooks as React hooks.
- For Lynx-specific hooks, cross-thread functions, and compile-time macros, assert only external behavior.
- For macros such as `__MAIN_THREAD__` / `__BACKGROUND__`, mock branch conditions according to the test environment.

## Layout and Style

Do not assert real layout dimensions. Assert class/style assignment, key element presence, and interactions.

## Common Errors

| Error | Fix |
|---|---|
| `lynx is not defined` | Mock global `lynx` in setup or test file |
| Native module undefined | Mock the corresponding NativeModules path |
| Element not found | Use queries supported by the Lynx test library; avoid HTML role assumptions |
| Thread/macro variable missing | Mock compile-time macros or test observable output |
